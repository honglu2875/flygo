"""Fixed local spherical sampling; geometry and visual encoding stay outside the CNS.

Directions supplied by a registered atlas can replace the engineered hex chart.
Neither soma coordinates nor missing photoreceptors estimate optical acuity.
"""
from dataclasses import dataclass
import numpy as np


VERSION = 'bilateral-spherical-luminance-h4-v1'


def directions(azimuth, elevation):
    """Radians, with forward +X, right +Y, up +Z; returns unit vectors."""
    a, e = np.broadcast_arrays(azimuth, elevation)
    return np.stack([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)], axis=-1)


def hex_chart(coordinates, *, origin, radians_per_column, reflect=False):
    """Engineered exponential chart, NOT a measured optical-axis registration.

    Declared 120-degree axial convention: (1,0), (0,1), (1,1) are neighbors.
    The origin/scale come from a common reference lattice, never per-eye holes.
    Reflection is an explicit alternative until anatomical orientation is known.
    """
    qr = np.asarray(coordinates, np.float64) - np.asarray(origin, np.float64)
    if qr.ndim != 2 or qr.shape[1] != 2 or not np.isfinite(qr).all():
        raise ValueError('Expected finite hex coordinates [cells,2]')
    if not np.isfinite(radians_per_column) or radians_per_column <= 0:
        raise ValueError('A positive finite angular chart scale is required')
    xy = np.column_stack([qr[:,0]-.5*qr[:,1], np.sqrt(3)/2*qr[:,1]]) * radians_per_column
    if reflect: xy[:,0] *= -1
    radius = np.linalg.norm(xy, axis=1)
    if np.any(radius >= np.pi/2):
        raise ValueError('Engineered chart must stay inside one open hemisphere')
    return np.column_stack([np.cos(radius), np.sinc(radius/np.pi)[:,None]*xy])


@dataclass(frozen=True)
class SphericalRenderer:
    """Two 9x9 patches per eye: L lags 0/2, R lags 1/3.

    Each receptor samples only its nearest patch with a compact angular kernel.
    Own/opponent/empty luminances are .95/.05/.5. Uncovered or missing history
    is neutral .5. All receptor classes at one qualified column see one image.
    Context planes are deliberately excluded from this representation pilot.
    """
    index: np.ndarray              # [sensors, neighbors], flattened [lag,point]
    weight: np.ndarray             # same shape; row sum <= 1, neutral elsewhere
    unit: np.ndarray
    side: np.ndarray
    patch: np.ndarray
    size: int = 9
    history: int = 4

    @classmethod
    def build(cls, unit, side, *, size=9, sigma_degrees=5.0, neighbors=4,
              patch_centers_degrees=((-25.,-30.),(-25.,30.)), halfwidth_degrees=25.):
        unit = np.asarray(unit, np.float64)
        side = np.asarray(side)
        if (unit.ndim != 2 or unit.shape[1] != 3 or len(side) != len(unit)
                or not np.isfinite(unit).all() or not np.allclose(np.linalg.norm(unit,axis=1),1,atol=1e-7)
                or not np.isin(side, ['L','R']).all() or size < 2 or not 1 <= neighbors <= size*size
                or not np.isfinite(sigma_degrees) or not 0 < sigma_degrees <= 15):
            raise ValueError('Invalid spherical renderer geometry')
        # Equal allocation first. No claim of a measured fovea or recency acuity.
        centers = np.asarray(patch_centers_degrees,np.float64)
        if centers.shape==(2,):centers=np.column_stack([centers,np.zeros(2)])
        if (centers.shape!=(2,2) or not np.isfinite(centers).all() or np.array_equal(centers[0],centers[1])
                or not np.isfinite(halfwidth_degrees) or halfwidth_degrees<=0
                or np.max(np.abs(centers))+halfwidth_degrees>=90):
            raise ValueError('Two distinct patch centers and a positive hemisphere-bounded halfwidth are required')
        centers = np.deg2rad(centers)
        grid = np.linspace(-np.deg2rad(halfwidth_degrees), np.deg2rad(halfwidth_degrees), size)
        row, col = np.meshgrid(grid, grid, indexing='ij')
        patch_center = directions(centers[:,0],centers[:,1])
        patch = np.argmax(unit @ patch_center.T, axis=1)
        index = np.empty((len(unit),neighbors),np.int32)
        weight = np.zeros((len(unit),neighbors),np.float32)
        sigma = np.deg2rad(sigma_degrees)
        for p in range(2):
            cells = np.flatnonzero(patch == p)
            points = directions(col.ravel()+centers[p,0], centers[p,1]-row.ravel())
            angle = np.arccos(np.clip(unit[cells] @ points.T, -1, 1))
            closest = np.argsort(angle, axis=1, kind='stable')[:,:neighbors]
            d = np.take_along_axis(angle, closest, axis=1)
            w = np.where(d <= 2*sigma, np.exp(-.5*(d/sigma)**2), 0.)
            w /= np.maximum(w.sum(axis=1,keepdims=True), 1.)
            lag = 2*p + (side[cells] == 'R').astype(np.int32)
            index[cells] = closest + lag[:,None]*size*size
            weight[cells] = w.astype(np.float32)
        return cls(index, weight, unit.copy(), side.copy(), patch.astype(np.int32), size)

    def render(self, features, *, contrast=1.0):
        x = np.asarray(features, np.float32)
        if x.ndim != 4 or x.shape[1:] != (self.size,self.size,2*self.history+4) or not np.isfinite(x).all():
            raise ValueError('Expected finite NHWC Go features with four historical boards')
        if np.any((x[...,:8] < 0) | (x[...,:8] > 1)):
            raise ValueError('Occupancy planes must lie in [0,1]')
        c = np.asarray(contrast, np.float32)
        if c.ndim > 1 or (c.ndim == 1 and len(c) != len(x)) or not np.isfinite(c).all() or np.any((c < .5) | (c > 1.1)):
            raise ValueError('Contrast must be scalar or per position, within [.5,1.1]')
        signed = (x[...,:8:2]-x[...,1:8:2]).transpose(0,3,1,2).reshape(len(x),-1)
        sampled = (signed[:,self.index]*self.weight).sum(axis=-1,dtype=np.float32)
        return np.float32(.5) + np.float32(.45)*c.reshape(-1,1)*sampled

    def audit(self):
        rows = []
        for eye in ('L','R'):
            for patch in range(2):
                use = (self.side == eye) & (self.patch == patch)
                idx, w = self.index[use], self.weight[use]
                mass = np.bincount((idx % self.size**2).ravel(), weights=w.ravel(), minlength=self.size**2)
                rows.append(dict(eye=eye, lag=2*patch+(eye=='R'), sensors=int(use.sum()),
                    illuminated_sensors=int(np.count_nonzero(w.sum(axis=1))),
                    covered_board_points=int(np.count_nonzero(mass)),
                    point_mass_min=float(mass.min()), point_mass_max=float(mass.max())))
        return rows

    def arrays(self):
        return {key:getattr(self,key) for key in ('index','weight','unit','side','patch')}


def individual_motor_ports(neurons, sensors, motors):
    """One input per qualified photoreceptor, one output per anatomical candidate."""
    sensors, motors = np.asarray(sensors), np.asarray(motors)
    if (len(np.unique(sensors)) != len(sensors) or len(np.unique(motors)) != len(motors)
            or len(np.intersect1d(sensors,motors)) or not len(sensors) or not len(motors)
            or min(sensors.min(),motors.min()) < 0 or max(sensors.max(),motors.max()) >= neurons):
        raise ValueError('Sensory and motor identities must be unique, disjoint and in range')
    index = np.full(neurons,-1,np.int32); group = index.copy(); scale = np.zeros(neurons,np.float32)
    index[sensors] = np.arange(len(sensors),dtype=np.int32)
    group[motors] = np.arange(len(motors),dtype=np.int32); scale[motors] = 1
    return dict(input_index=index,output_group=group,output_scale=scale)
