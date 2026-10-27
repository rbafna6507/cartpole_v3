"""Generate a two-DOF, contact-free MJCF surrogate from explicit configuration."""
from .config import SimulationConfig


def model_xml(config: SimulationConfig) -> str:
    # A slender uniform rod of length l has pivot inertia m*l²/3 and COM l/2.
    # I_pivot/(m*COM)=2*l/3, so l=1.5*effective_length matches L_eff exactly.
    length = 1.5 * config.effective_length_m
    transverse_inertia = config.pole_mass_kg * length**2 / 12
    # Positive axial inertia is required by MuJoCo. It has no effect about +y.
    axial_inertia = transverse_inertia * 1e-6
    return f'''<mujoco model="provisional_cartpole_uniform_rod">
  <compiler angle="radian" inertiafromgeom="false"/>
  <option timestep="{config.physics_dt_ns / 1e9:.17g}" gravity="0 0 -9.81" integrator="Euler">
    <flag eulerdamp="disable" autoreset="disable"/>
  </option>
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <body name="cart">
      <joint name="cart_slide" type="slide" axis="1 0 0" damping="{config.cart_damping:.17g}" limited="false"/>
      <inertial pos="0 0 0" mass="{config.cart_mass_kg:.17g}" diaginertia=".001 .001 .001"/>
      <geom type="box" size=".04 .025 .02" rgba=".2 .3 .6 1"/>
      <body name="pole">
        <joint name="pole_hinge" type="hinge" axis="0 1 0" damping="{config.hinge_damping:.17g}" limited="false"/>
        <inertial pos="0 0 {length/2:.17g}" mass="{config.pole_mass_kg:.17g}"
          diaginertia="{transverse_inertia:.17g} {transverse_inertia:.17g} {axial_inertia:.17g}"/>
        <geom type="capsule" fromto="0 0 0 0 0 {length:.17g}" size=".004" rgba=".8 .3 .2 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>'''
