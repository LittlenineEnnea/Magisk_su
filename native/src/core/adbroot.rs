// Run adbd as root without ever exposing a global `ro.debuggable=1`.
//
// adbd decides whether to drop privileges exactly once, at startup, from
// `ro.secure`, `ro.debuggable` and `service.adb.root`. Rather than flipping
// those properties globally, magiskd starts adbd itself inside a private mount
// namespace in which the backing files of the property area that hold those
// properties are replaced by patched copies. Only adbd observes the patched
// values; every other process keeps reading the untouched global property
// area, so `getprop ro.debuggable` stays 0 system wide.
//
// Why magiskd parents adbd instead of letting init start it:
//
//   `adb root` makes adbd set `service.adb.root=1` and exit, expecting its
//   parent to bring it back. If init owns adbd, the replacement is started
//   outside our namespace and reads the real `ro.debuggable=0`, so it drops
//   privileges again. Parenting adbd to magiskd is what keeps the private
//   property view across an `adb root`.
//
// Why the namespace work happens before execve rather than by ptracing adbd:
//
//   `unshare(CLONE_NEWNS)` needs CAP_SYS_ADMIN, and SELinux rules cannot grant
//   a capability. Between fork() and execve() the child is still magiskd:
//   uid 0, full capabilities, in the (permissive) magisk domain. The SELinux
//   transition into u:r:adbd:s0 is deferred to execve through
//   /proc/self/attr/exec, so adbd still runs in its own domain and its own
//   `--root_seclabel` transition keeps working.
//
// Property area layout: short values live inline in the `prop_info` record
// that precedes the property name:
//
//     struct prop_info {
//         atomic_uint_least32_t serial;  // offset 0, 4 bytes
//         char value[PROP_VALUE_MAX];    // offset 4, 92 bytes
//         char name[0];                  // offset 96
//     };
//
// `serial >> 24` is the value length, so replacing a one byte value with
// another one byte value needs no serial fixup and no trie surgery. Both
// properties we care about go "0" -> "1", so a plain byte patch is enough.
//
// `service.adb.root` may not exist yet at boot, and a property that does not
// exist cannot be byte patched into existence. It is therefore created once
// through property_service and immediately reset to "0" globally; the entry
// then exists with a one byte value and the private copy can patch it. Global
// state is left exactly as it was found.
//
// Nothing here runs in init or in first stage and no .rc file is touched. The
// worst case failure mode is that adbd does not come back until the next
// reboot; this can never cause a boot loop.

use base::{LoggedResult, ResultExt, Utf8CStr, cstr, info, libc, warn};
use std::ffi::{CString, c_char};
use std::fs;
use std::os::fd::RawFd;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use crate::resetprop::get_prop;

/// Offset of `prop_info::value` relative to the start of the record.
const PROP_INFO_VALUE_OFFSET: usize = 4;
/// Offset of `prop_info::name` relative to the start of the record.
const PROP_INFO_NAME_OFFSET: usize = 96;

const PROP_DIR: &str = "/dev/__properties__";
const STAGE_DIR: &str = "/dev/.magisk_adbroot";
const ADBD_APEX_PATH: &str = "/apex/com.android.adbd/bin/adbd";
const ADBD_SYSTEM_PATH: &str = "/system/bin/adbd";
const ADBD_SOCKET: &str = "/dev/socket/adbd";

const ADBD_CON: &Utf8CStr = cstr!("u:r:adbd:s0");
const ADBD_SOCKET_CON: &Utf8CStr = cstr!("u:object_r:adbd_socket:s0");

/// Properties that have to read as "1" from inside adbd's namespace.
const PATCH_PROPS: [&str; 2] = ["ro.debuggable", "service.adb.root"];

/// Child setup failure is reported through the exit status, because nothing
/// between fork() and execve() may touch magiskd's logger.
const EXIT_UNSHARE: i32 = 91;
const EXIT_SLAVE: i32 = 92;
const EXIT_BIND: i32 = 93;
const EXIT_SETEXECCON: i32 = 94;
const EXIT_EXEC: i32 = 96;

/// True while magiskd, not init, owns the adbd service.
static OWN_ADBD: AtomicBool = AtomicBool::new(false);

unsafe extern "C" {
    /// The real property setter, which goes through init's property_service.
    /// Magisk's own `set_prop` deliberately skips property_service, so it
    /// cannot be used for `ctl.*` nor to create a property that is missing.
    fn __system_property_set(key: *const c_char, value: *const c_char) -> i32;
}

/// Everything the child needs, fully built before fork(): no allocation runs
/// in the child, because malloc's lock may be held by another magiskd thread
/// at the moment we fork.
struct SpawnPlan {
    /// adbd binary, and argv/envp pointer arrays kept alive by `_argv`/`_envp`.
    exec_path: CString,
    _argv: Vec<CString>,
    _envp: Vec<CString>,
    argv: Vec<*const c_char>,
    envp: Vec<*const c_char>,
    /// (original property file, patched copy) pairs to bind mount.
    mounts: Vec<(CString, CString)>,
    /// Inherited control socket, or -1.
    socket_fd: RawFd,
}

// The raw pointers only ever point into this struct's own CStrings.
unsafe impl Send for SpawnPlan {}

fn prop_service_set(key: &str, val: &str) -> bool {
    let (Ok(k), Ok(v)) = (CString::new(key), CString::new(val)) else {
        return false;
    };
    unsafe { __system_property_set(k.as_ptr(), v.as_ptr()) == 0 }
}

fn prop(key: &str) -> String {
    let mut k = key.to_owned();
    get_prop(Utf8CStr::from_string(&mut k))
}

fn adbd_running() -> bool {
    prop("init.svc.adbd") == "running"
}

fn adbd_binary() -> Option<&'static str> {
    if Path::new(ADBD_APEX_PATH).exists() {
        Some(ADBD_APEX_PATH)
    } else if Path::new(ADBD_SYSTEM_PATH).exists() {
        Some(ADBD_SYSTEM_PATH)
    } else {
        None
    }
}

/// Locate `name` in a property area image and return the offset of the inline
/// value of its `prop_info` record.
fn find_value_offset(buf: &[u8], name: &str) -> Option<usize> {
    let mut needle = Vec::with_capacity(name.len() + 1);
    needle.extend_from_slice(name.as_bytes());
    needle.push(0);

    let mut from = 0usize;
    while from + needle.len() <= buf.len() {
        let Some(rel) = buf[from..]
            .windows(needle.len())
            .position(|w| w == needle.as_slice())
        else {
            return None;
        };
        let pos = from + rel;
        if pos >= PROP_INFO_NAME_OFFSET {
            let base = pos - PROP_INFO_NAME_OFFSET;
            let value = base + PROP_INFO_VALUE_OFFSET;
            if value + 2 <= buf.len() {
                let serial =
                    u32::from_le_bytes([buf[base], buf[base + 1], buf[base + 2], buf[base + 3]]);
                // Only a one byte, NUL terminated value can be patched without
                // rewriting the serial length field.
                if (serial >> 24) == 1 && buf[value + 1] == 0 {
                    return Some(value);
                }
            }
        }
        from = pos + 1;
    }
    None
}

/// Mirror the SELinux label of `src` onto `dst` through the raw xattr, so this
/// does not depend on libselinux being linked into magiskd.
fn copy_secontext(src: &Path, dst: &Path) {
    let (Ok(src_c), Ok(dst_c)) = (
        CString::new(src.as_os_str().as_bytes()),
        CString::new(dst.as_os_str().as_bytes()),
    ) else {
        return;
    };
    let attr = cstr!("security.selinux");
    let mut buf = [0u8; 256];
    let len = unsafe {
        libc::getxattr(
            src_c.as_ptr(),
            attr.as_ptr(),
            buf.as_mut_ptr().cast(),
            buf.len(),
        )
    };
    if len <= 0 {
        return;
    }
    unsafe {
        libc::setxattr(
            dst_c.as_ptr(),
            attr.as_ptr(),
            buf.as_ptr().cast(),
            len as usize,
            0,
        );
    }
}

/// Copy every property area file holding one of `PATCH_PROPS` into the staging
/// directory with the value byte flipped to '1'.
fn stage_patched_props() -> LoggedResult<Vec<(CString, CString)>> {
    let stage = Path::new(STAGE_DIR);
    let _ = fs::remove_dir_all(stage);
    fs::create_dir_all(stage)?;
    if let Ok(dir_c) = CString::new(STAGE_DIR) {
        unsafe { libc::chmod(dir_c.as_ptr(), 0o700) };
    }

    let mut out: Vec<(CString, CString)> = Vec::new();

    for entry in fs::read_dir(PROP_DIR)? {
        let entry = entry?;
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();
        let Ok(mut buf) = fs::read(&path) else {
            continue;
        };

        let mut patched = false;
        for name in PATCH_PROPS {
            if let Some(off) = find_value_offset(&buf, name) {
                buf[off] = b'1';
                patched = true;
                info!("adbroot: patched {} in {}", name, path.display());
            }
        }
        if !patched {
            continue;
        }

        let Some(file_name) = path.file_name() else {
            continue;
        };
        let dst: PathBuf = stage.join(file_name);
        fs::write(&dst, &buf)?;
        if let Ok(dst_c) = CString::new(dst.as_os_str().as_bytes()) {
            unsafe { libc::chmod(dst_c.as_ptr(), 0o444) };
        }
        // adbd opens these read only in its own domain, so the copy has to
        // carry the same label as the file it shadows.
        copy_secontext(&path, &dst);

        let (Ok(orig_c), Ok(dst_c)) = (
            CString::new(path.as_os_str().as_bytes()),
            CString::new(dst.as_os_str().as_bytes()),
        ) else {
            continue;
        };
        out.push((orig_c, dst_c));
    }

    if out.is_empty() {
        warn!("adbroot: none of the property files carried the target properties");
    }
    Ok(out)
}

/// Make sure each property we byte patch exists globally, so the copy has a
/// record to patch. Anything created here is reset to "0" right away.
fn ensure_props_exist() {
    for name in PATCH_PROPS {
        if prop(name).is_empty() {
            info!("adbroot: creating {} so it can be patched per-process", name);
            prop_service_set(name, "1");
            thread::sleep(Duration::from_millis(50));
            prop_service_set(name, "0");
            thread::sleep(Duration::from_millis(50));
        }
    }
}

fn write_attr(attr: &str, val: &[u8]) -> bool {
    fs::write(format!("/proc/self/attr/{attr}"), val).is_ok()
}

/// Recreate the seqpacket control socket init would normally hand to adbd, so
/// that AdbDebuggingManager can still reach it. Best effort: adbd runs without
/// it, it only loses the authorization prompt.
fn create_adbd_socket() -> RawFd {
    let _ = fs::remove_file(ADBD_SOCKET);
    // The socket node must carry adbd's socket label, not magisk's.
    write_attr("fscreate", ADBD_SOCKET_CON.as_bytes());

    let fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_SEQPACKET, 0) };
    let result = (|| -> Option<RawFd> {
        if fd < 0 {
            return None;
        }
        let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
        addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
        let bytes = ADBD_SOCKET.as_bytes();
        if bytes.len() >= addr.sun_path.len() {
            return None;
        }
        for (dst, src) in addr.sun_path.iter_mut().zip(bytes) {
            *dst = *src as c_char;
        }
        let len = (std::mem::size_of::<libc::sa_family_t>() + bytes.len() + 1) as libc::socklen_t;
        let addr_ptr = &addr as *const libc::sockaddr_un as *const libc::sockaddr;
        if unsafe { libc::bind(fd, addr_ptr, len) } != 0 {
            return None;
        }
        if unsafe { libc::listen(fd, 4) } != 0 {
            return None;
        }
        Some(fd)
    })();

    write_attr("fscreate", b"");

    match result {
        Some(fd) => {
            if let Ok(path_c) = CString::new(ADBD_SOCKET) {
                unsafe {
                    libc::chmod(path_c.as_ptr(), 0o660);
                    // AID_SYSTEM / AID_SYSTEM, as in adbd.rc
                    libc::chown(path_c.as_ptr(), 1000, 1000);
                }
            }
            fd
        }
        None => {
            warn!("adbroot: could not create {}", ADBD_SOCKET);
            if fd >= 0 {
                unsafe { libc::close(fd) };
            }
            -1
        }
    }
}

fn build_plan(bin: &str, mounts: Vec<(CString, CString)>, socket_fd: RawFd) -> Option<SpawnPlan> {
    let exec_path = CString::new(bin).ok()?;

    let argv_owned = vec![
        CString::new(bin).ok()?,
        CString::new("--root_seclabel=u:r:su:s0").ok()?,
    ];

    // Mirror the environment init gives its services.
    let mut envp_owned = vec![
        CString::new(
            "PATH=/product/bin:/apex/com.android.runtime/bin:/system/bin:\
             /system_ext/bin:/odm/bin:/vendor/bin:/vendor/xbin",
        )
        .ok()?,
        CString::new("ANDROID_DATA=/data").ok()?,
        CString::new("ANDROID_ROOT=/system").ok()?,
    ];
    if socket_fd >= 0 {
        envp_owned.push(CString::new(format!("ANDROID_SOCKET_adbd={socket_fd}")).ok()?);
    }

    let argv: Vec<*const c_char> = argv_owned
        .iter()
        .map(|s| s.as_ptr())
        .chain(std::iter::once(ptr::null()))
        .collect();
    let envp: Vec<*const c_char> = envp_owned
        .iter()
        .map(|s| s.as_ptr())
        .chain(std::iter::once(ptr::null()))
        .collect();

    Some(SpawnPlan {
        exec_path,
        _argv: argv_owned,
        _envp: envp_owned,
        argv,
        envp,
        mounts,
        socket_fd,
    })
}

/// Runs between fork() and execve(). No allocation, no logging, no locks.
unsafe fn child_exec(plan: &SpawnPlan) -> ! {
    unsafe {
        // 1. Private mount namespace, so the bind mounts stay invisible.
        if libc::unshare(libc::CLONE_NEWNS) != 0 {
            libc::_exit(EXIT_UNSHARE);
        }

        // 2. `/` is normally rshared, which would propagate the bind mounts
        //    straight back into init's namespace. Cut propagation first.
        if libc::mount(
            ptr::null(),
            c"/".as_ptr(),
            ptr::null(),
            libc::MS_REC | libc::MS_SLAVE,
            ptr::null(),
        ) != 0
        {
            libc::_exit(EXIT_SLAVE);
        }

        // 3. Swap each property area file for its patched copy.
        for (orig, patched) in &plan.mounts {
            if libc::mount(
                patched.as_ptr(),
                orig.as_ptr(),
                ptr::null(),
                libc::MS_BIND,
                ptr::null(),
            ) != 0
            {
                libc::_exit(EXIT_BIND);
            }
        }

        // 4. Transition into adbd's domain at execve. Running adbd as magisk
        //    makes its own --root_seclabel setcon fail and adbd aborts.
        let fd = libc::open(
            c"/proc/self/attr/exec".as_ptr(),
            libc::O_WRONLY | libc::O_CLOEXEC,
        );
        if fd < 0 {
            libc::_exit(EXIT_SETEXECCON);
        }
        let con = ADBD_CON.as_bytes_with_nul();
        let n = libc::write(fd, con.as_ptr().cast(), con.len());
        libc::close(fd);
        if n < 0 {
            libc::_exit(EXIT_SETEXECCON);
        }

        // 5. Let adbd inherit the control socket.
        if plan.socket_fd >= 0 {
            let flags = libc::fcntl(plan.socket_fd, libc::F_GETFD);
            if flags >= 0 {
                libc::fcntl(plan.socket_fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC);
            }
        }

        libc::execve(
            plan.exec_path.as_ptr(),
            plan.argv.as_ptr(),
            plan.envp.as_ptr(),
        );
        libc::_exit(EXIT_EXEC);
    }
}

fn spawn_rooted_adbd(plan: &SpawnPlan) -> Option<libc::pid_t> {
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        warn!("adbroot: fork failed");
        return None;
    }
    if pid == 0 {
        unsafe { child_exec(plan) }
    }
    Some(pid)
}

/// Ask init to stop its own adbd and wait for it to go away.
fn stop_init_adbd() {
    prop_service_set("ctl.stop", "adbd");
    for _ in 0..50 {
        if !adbd_running() {
            return;
        }
        thread::sleep(Duration::from_millis(100));
    }
    warn!("adbroot: init.svc.adbd never reached stopped");
}

/// init restarts its own adbd from USB property triggers
/// (`sys.usb.config`, `persist.sys.usb.config`). Two adbd instances would
/// fight over the USB endpoint, so keep init's one down for as long as we own
/// adbd.
fn spawn_init_adbd_watchdog() {
    OWN_ADBD.store(true, Ordering::SeqCst);
    thread::Builder::new()
        .name("adbroot-wd".into())
        .spawn(|| {
            while OWN_ADBD.load(Ordering::SeqCst) {
                thread::sleep(Duration::from_secs(1));
                if OWN_ADBD.load(Ordering::SeqCst) && adbd_running() {
                    info!("adbroot: init restarted its adbd, stopping it again");
                    prop_service_set("ctl.stop", "adbd");
                }
            }
        })
        .log_ok();
}

/// Hand adbd back to init: stop shadowing its service, then start it.
fn release_to_init() {
    OWN_ADBD.store(false, Ordering::SeqCst);
    // Let the watchdog observe the flag before init's instance comes up.
    thread::sleep(Duration::from_millis(1200));
    prop_service_set("ctl.start", "adbd");
}

/// Keep a rooted adbd alive. adbd exits on `adb root`; restarting it here is
/// what preserves the private property view across that restart.
fn supervise(plan: SpawnPlan) {
    let mut failures = 0;
    spawn_init_adbd_watchdog();

    loop {
        let Some(pid) = spawn_rooted_adbd(&plan) else {
            release_to_init();
            return;
        };
        info!("adbroot: started rooted adbd, pid={}", pid);

        let mut status: libc::c_int = 0;
        let reaped = unsafe { libc::waitpid(pid, &mut status, 0) };

        let code = if reaped == pid {
            if libc::WIFEXITED(status) {
                libc::WEXITSTATUS(status)
            } else {
                -1
            }
        } else {
            // Magisk reaps children with waitpid(-1) while running module
            // scripts, so our child may already be gone. Wait for the pid to
            // actually disappear rather than treating this as fatal.
            warn!("adbroot: pid {} was reaped elsewhere, polling instead", pid);
            while unsafe { libc::kill(pid, 0) } == 0 {
                thread::sleep(Duration::from_millis(200));
            }
            -1
        };

        if (EXIT_UNSHARE..=EXIT_EXEC).contains(&code) {
            warn!("adbroot: child setup failed at stage {}, aborting", code);
            release_to_init();
            return;
        }

        // `adb unroot` sets service.adb.root=0 globally and exits. Honour that
        // by handing adbd back to init, unrooted.
        if prop("service.adb.root") == "0" && code == 0 {
            info!("adbroot: unroot requested, handing adbd back to init");
            release_to_init();
            return;
        }

        // An adbd that dies immediately and repeatedly must not be respawned
        // forever; give ADB back to init instead.
        if code != 0 {
            failures += 1;
            if failures >= 5 {
                warn!("adbroot: adbd keeps failing (code {}), giving up", code);
                release_to_init();
                return;
            }
        } else {
            failures = 0;
        }

        info!("adbroot: adbd exited (code {}), restarting", code);
        thread::sleep(Duration::from_millis(300));
    }
}

/// Entry point, called from `boot_complete`.
pub fn setup_adb_root() {
    // Never force ADB on: if adbd is not running, ADB is off and we stay out.
    if !adbd_running() {
        info!("adbroot: adbd is not running, leaving ADB untouched");
        return;
    }
    let Some(bin) = adbd_binary() else {
        warn!("adbroot: no adbd binary found");
        return;
    };

    ensure_props_exist();

    let mounts = match stage_patched_props() {
        Ok(m) if !m.is_empty() => m,
        _ => {
            warn!("adbroot: could not stage patched property files");
            return;
        }
    };

    // Stop init's instance before touching the socket node it currently owns.
    stop_init_adbd();

    let socket_fd = create_adbd_socket();
    let Some(plan) = build_plan(bin, mounts, socket_fd) else {
        warn!("adbroot: could not build the spawn plan");
        prop_service_set("ctl.start", "adbd");
        return;
    };

    thread::Builder::new()
        .name("adbroot".into())
        .spawn(move || supervise(plan))
        .log_ok();
}
