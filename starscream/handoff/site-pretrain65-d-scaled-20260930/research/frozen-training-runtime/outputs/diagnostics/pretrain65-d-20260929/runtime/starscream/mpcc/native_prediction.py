"""Small C99 prediction binding. Opt-in; never silently changes backend.

Build once per source/compiler/flag digest with a process lock. ctypes releases
the GIL during the C call. Source and NumPy reference remain versioned together.
"""
import ctypes
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess
import tempfile
import os
import numpy as np


@lru_cache(maxsize=1)
def library():
    import fcntl
    source = Path(__file__).with_suffix('.c')
    flags = ['-O3','-std=c99','-shared','-fPIC','-fno-fast-math','-ffp-contract=off']
    version = subprocess.check_output(['cc','--version'])
    digest = hashlib.sha256(source.read_bytes()+version+repr(flags).encode()).hexdigest()[:20]
    root = Path(tempfile.gettempdir()) / f'starscream-native-{os.getuid()}'
    root.mkdir(exist_ok=True, mode=0o700)
    target = root / f'prediction-{digest}.so'
    with (root / f'{digest}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            temporary = root / f'{digest}-{os.getpid()}.tmp.so'
            subprocess.run(['cc',*flags,str(source),'-lm','-o',str(temporary)],check=True)
            temporary.replace(target)
    lib = ctypes.CDLL(str(target))
    array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    lib.starscream_predict_step.argtypes = [array,array,array,ctypes.c_double,array]
    lib.starscream_predict_step.restype = None
    lib.starscream_rk4.argtypes = [array,ctypes.c_double,array]
    lib.starscream_rk4.restype = None
    lib.starscream_corridor_update.argtypes = [ctypes.c_void_p]*5+[ctypes.c_int,array,array]
    lib.starscream_corridor_update.restype = None
    lib.starscream_cross.argtypes = [array,array,array]
    lib.starscream_cross.restype = None
    lib.starscream_reference_progress.argtypes = [array,array,array,ctypes.c_int,array,array,ctypes.c_int,array]
    lib.starscream_reference_progress.restype = None
    lib.starscream_cubic_triplet.argtypes = [array, array, ctypes.c_int, array, ctypes.c_int, array]
    lib.starscream_cubic_triplet.restype = None
    lib.starscream_project_progress.argtypes = [array, array, array, ctypes.c_int,
        ctypes.c_double, ctypes.c_int, array, ctypes.c_double, ctypes.c_int,
        ctypes.c_double, ctypes.c_void_p, ctypes.c_int]
    lib.starscream_project_progress.restype = ctypes.c_double
    lib.starscream_predict_step_exact.argtypes = [array, array, array, array, ctypes.c_int,
        array, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.starscream_predict_step_exact.restype = ctypes.c_int
    return lib


class NativeCubicTriplet:
    """One C crossing for three evaluations of the SAME SciPy coefficients.

    Own the spline, not a copied approximation. Coefficient mutation stays
    visible; replacing the spline requires rebuilding this wrapper.
    """
    def __init__(self, spline):
        self.spline = spline
        if (spline.c.shape != (4, len(spline.x)-1, 3)
                or spline.c.dtype != np.float64 or spline.x.dtype != np.float64
                or not spline.c.flags.c_contiguous or not spline.x.flags.c_contiguous
                or np.any(np.diff(spline.x) <= 0)):
            raise ValueError('native cubic requires contiguous FP64 3D cubic coefficients')
        self.function = library().starscream_cubic_triplet
        self.knots, self.coefficients = spline.x, spline.c

    def __call__(self, query):
        if self.spline.x is not self.knots or self.spline.c is not self.coefficients:
            raise ValueError('spline buffers replaced: rebuild the native wrapper')
        q = np.asarray(query, np.float64)
        values = np.ascontiguousarray(q).reshape(-1)
        output = np.empty((3, len(values), 3), np.float64)
        self.function(self.spline.x, self.spline.c.reshape(-1), len(self.spline.x)-1,
                      values, len(values), output.reshape(-1))
        return output.reshape((3,) + q.shape + (3,))


@lru_cache(maxsize=1)
def numpy_blas_dot():
    # Resolve through NumPy's extension dependency tree, never an arbitrary
    # system BLAS with different rounding/threading. Refuse unsupported builds.
    from numpy._core import _multiarray_umath
    lib = ctypes.CDLL(_multiarray_umath.__file__)
    for name, wide in [('cblas_ddot64_', True), ('cblas_ddot', False)]:
        try:
            fn = getattr(lib, name)
        except AttributeError:
            continue
        integer = ctypes.c_int64 if wide else ctypes.c_int
        array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        fn.argtypes = [integer, array, integer, array, integer]
        fn.restype = ctypes.c_double
        rng = np.random.default_rng(82)
        for _ in range(128):
            a, b = rng.normal(size=(2, 3))
            if fn(3, a, 1, b, 1) != float(a @ b):
                raise RuntimeError('native projection BLAS does not match NumPy exactly')
        return lib, fn, wide
    raise RuntimeError('native projection requires a supported NumPy CBLAS symbol')


class NativeProgressProjection:
    def __init__(self, line):
        self.line = line
        self.blas, self.dot, self.wide = numpy_blas_dot()
        self.pointer = ctypes.cast(self.dot, ctypes.c_void_p)
        self.function = library().starscream_project_progress
        # Read-only coefficient reference, same contract as the triplet wrapper.
        self.coefficients = NativeCubicTriplet(line._position_spline)
        self.samples = len(line.progress)-1

    def __call__(self, point, hint, radius):
        line = self.line
        point = np.ascontiguousarray(point, np.float64)
        if (point.shape != (3,) or not np.isfinite(point).all()
                or not np.isfinite(radius) or (hint is not None and not np.isfinite(hint))):
            raise ValueError('native progress projection requires finite inputs')
        if (line.progress.shape != (self.samples+1,) or line.position.shape != (self.samples+1,3)
                or self.coefficients.spline.c is not self.coefficients.coefficients
                or self.coefficients.spline.c.shape != (4,self.samples,3)):
            raise ValueError('racing line buffers changed: rebuild native projection')
        return self.function(np.ascontiguousarray(line.progress),
            np.ascontiguousarray(line.position).reshape(-1),
            self.coefficients.spline.c.reshape(-1), len(line.progress)-1,
            line.length, int(line.loop), point, 0. if hint is None else hint,
            int(hint is not None), radius, self.pointer, int(self.wide))


class NativeCorridorUpdate:
    """Batch only this backend's known two-dimensional corridor bounds.

    First call goes through acados' dimension-checked public API at every stage.
    The installed acados ABI is checked; no numerical constraints are dropped.
    """
    def __init__(self, solver, horizon):
        self.solver=solver;self.horizon=horizon;self.checked=False
        lib=getattr(solver,'_AcadosOcpSolver__acados_lib')
        self.setter=lib.ocp_nlp_constraints_model_set
        if len(self.setter.argtypes)!=7:
            raise RuntimeError('unsupported acados constraint setter ABI')
        self.function=library().starscream_corridor_update

    def __call__(self, lower, upper):
        expected=(self.horizon+1,2)
        if lower.shape!=expected or upper.shape!=expected or not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError('invalid native corridor bounds')
        if not self.checked:
            for i in range(1,self.horizon):
                self.solver.constraints_set(i,'lh',lower[i]);self.solver.constraints_set(i,'uh',upper[i])
            self.checked=True
            return
        solver=self.solver
        self.function(ctypes.cast(self.setter,ctypes.c_void_p),solver.nlp_config,solver.nlp_dims,
            solver.nlp_in,solver.nlp_out,self.horizon,
            np.ascontiguousarray(lower).reshape(-1),np.ascontiguousarray(upper).reshape(-1))


class NativePrediction:
    def __init__(self, model):
        c=model.config
        self.parameters=np.concatenate([
            [c.mass,c.gravity,c.integration_dt_max,c.motor_tau,c.motor_omega_min,c.motor_omega_max,0.],
            model.inertia,model.rate_gain,model.body_rate_max,model.thrust_map,
            model.allocation.ravel(),model.allocation_inverse.ravel(),c.wind_world,c.linear_drag,
            c.quadratic_drag,c.rotor_drag,c.center_of_mass,c.angular_drag,[model.thrust_max]
        ]).astype(np.float64)
        assert self.parameters.shape == (70,)
        self.function=library().starscream_predict_step
        self.rk4_function=library().starscream_rk4
        self.inertia=np.ascontiguousarray(model.inertia,dtype=np.float64)
        self.cross_function=library().starscream_cross
        self.exact_schedule_cache = {}
        self.exact_blas = None

    def step_exact(self, state, motors, action, dt):
        state = np.array(state, dtype=np.float64, copy=True, order='C')
        motors = np.array(motors, dtype=np.float64, copy=True, order='C')
        action = np.ascontiguousarray(action, dtype=np.float64)
        if (state.shape != (25,) or motors.shape != (4,) or action.shape != (4,)
                or not np.isfinite(dt) or dt <= 0):
            raise ValueError('invalid exact native prediction inputs')
        if self.exact_blas is None:
            lib, dot, wide = numpy_blas_dot()
            suffix = '64_' if wide else ''
            self.exact_blas = (lib, dot, getattr(lib, 'cblas_sdot'+suffix),
                               getattr(lib, 'cblas_dgemv'+suffix), wide)
        if dt not in self.exact_schedule_cache:
            remaining, maximum = np.float32(dt), np.float32(self.parameters[2])
            schedule = []
            while remaining > 0.:
                h = float(np.minimum(remaining, maximum))
                schedule.append((h, float(np.exp(-h/self.parameters[3]))))
                remaining = np.float32(remaining - np.float32(h))
            if len(self.exact_schedule_cache) >= 16:
                self.exact_schedule_cache.clear()
            self.exact_schedule_cache[dt] = np.asarray(schedule, np.float64).reshape(-1)
        schedule = self.exact_schedule_cache[dt]
        lib, dot, sdot, gemv, wide = self.exact_blas
        status = library().starscream_predict_step_exact(state, motors, action, schedule,
            len(schedule)//2, self.parameters, ctypes.cast(dot, ctypes.c_void_p),
            ctypes.cast(sdot, ctypes.c_void_p), ctypes.cast(gemv, ctypes.c_void_p), int(wide))
        if status:
            raise ValueError('quaternion must be finite and nonzero')
        return state, motors

    def cross(self,left,right):
        left=np.ascontiguousarray(left,dtype=np.float64)
        right=np.ascontiguousarray(right,dtype=np.float64)
        if left.shape!=(3,) or right.shape!=(3,):
            raise ValueError('native cross requires two three-vectors')
        result=np.empty(3,np.float64)
        self.cross_function(left,right,result)
        return result

    def reference_progress(self,progress,speed,dt,xp,fp,options):
        dt=np.ascontiguousarray(dt,dtype=np.float64)
        xp=np.ascontiguousarray(xp,dtype=np.float64)
        fp=np.ascontiguousarray(fp,dtype=np.float64)
        options=np.ascontiguousarray(options,dtype=np.float64)
        # Validate buffer bounds before entering C. Outputs are intentionally
        # in-place: never silently copy a noncontiguous caller's output array.
        if (dt.ndim!=1 or xp.ndim!=1 or len(xp)<2 or fp.shape!=xp.shape
                or options.shape!=(6,)
                or any(not isinstance(v,np.ndarray) or v.dtype!=np.float64
                    or v.shape!=(len(dt)+1,) or not v.flags.c_contiguous
                    or not v.flags.writeable for v in (progress,speed))):
            raise ValueError('invalid native reference buffers')
        library().starscream_reference_progress(progress,speed,dt,len(dt),xp,fp,len(xp),options)

    def rk4(self,state,dt):
        state=np.array(state,dtype=np.float64,copy=True,order='C')
        if state.shape!=(25,) or not np.isfinite(state).all() or not np.isfinite(dt) or dt<=0:
            raise ValueError('invalid native RK4 inputs')
        self.rk4_function(state,dt,self.inertia)
        return state

    def step(self, state, motors, action, dt):
        state=np.array(state,dtype=np.float64,copy=True,order='C')
        motors=np.array(motors,dtype=np.float64,copy=True,order='C')
        action=np.ascontiguousarray(action,dtype=np.float64)
        if state.shape!=(25,) or motors.shape!=(4,) or action.shape!=(4,) or not np.isfinite(dt) or dt<=0:
            raise ValueError('invalid native prediction inputs')
        self.function(state,motors,action,dt,self.parameters)
        return state,motors
