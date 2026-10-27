"""Child-only Linux restrictions, installed before any uploaded source import."""
import ctypes
import errno
import math
import resource
import sys

# Names resolved by libseccomp on the actual architecture; no syscall numbers.
# pread64 is the positioned equivalent of read/lseek, needed by the ELF loader
# for explicitly mounted numeric libraries; it grants no additional path access.
SYSCALLS=('read','pread64','write','readv','writev','close','lseek','fstat','newfstatat','stat','lstat','statx','getdents64',
          'access','faccessat','faccessat2','open','openat','readlink','readlinkat','mmap','mmap2','mprotect',
          'munmap','mremap','brk','madvise','rt_sigaction','rt_sigprocmask','rt_sigreturn','sigaltstack',
          'futex','futex_time64','sched_yield','clock_gettime','clock_gettime64','gettimeofday','time',
          'nanosleep','clock_nanosleep','clock_nanosleep_time64','getrandom','getpid','gettid','getuid',
          'geteuid','getgid','getegid','uname','getrlimit','prlimit64','fcntl','fcntl64','ioctl','dup','dup2',
          'dup3','poll','ppoll','ppoll_time64','select','pselect6','pselect6_time64','exit','exit_group',
          'getcwd','fchdir','chdir','mkdir','mkdirat','unlink','unlinkat','rename','renameat','renameat2',
          'ftruncate','truncate','fsync','fdatasync','fchmod','chmod','umask','statfs','fstatfs','utimensat')

def install_limits(wall_seconds=60,memory_bytes=512*1024*1024):
    if sys.platform!='linux':raise RuntimeError('Linux sandbox required')
    for kind,value in ((resource.RLIMIT_CPU,max(1,math.ceil(wall_seconds+2))),
                       (resource.RLIMIT_AS,memory_bytes),(resource.RLIMIT_FSIZE,1024*1024),
                       (resource.RLIMIT_NOFILE,32),(resource.RLIMIT_NPROC,0),(resource.RLIMIT_CORE,0)):
        resource.setrlimit(kind,(value,value))
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(38,1,0,0,0)!=0:raise RuntimeError('no_new_privs failed')
    library=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    library.seccomp_init.argtypes=[ctypes.c_uint32];library.seccomp_init.restype=ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];library.seccomp_syscall_resolve_name.restype=ctypes.c_int
    library.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    library.seccomp_load.argtypes=[ctypes.c_void_p]
    library.seccomp_release.argtypes=[ctypes.c_void_p]
    context=library.seccomp_init(0x00050000|errno.EPERM)  # SCMP_ACT_ERRNO
    if not context:raise RuntimeError('seccomp allocation failed')
    try:
        for syscall in SYSCALLS:
            number=library.seccomp_syscall_resolve_name(syscall.encode())
            if number>=0 and library.seccomp_rule_add(context,0x7fff0000,number,0)!=0:
                raise RuntimeError('seccomp rule failed')
        if library.seccomp_load(context)!=0:raise RuntimeError('seccomp load failed')
    finally:library.seccomp_release(context)
