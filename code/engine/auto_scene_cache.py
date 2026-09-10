"""Bounded, batch-owned read-only scene reuse with input-family invalidation."""
from collections import OrderedDict
from pathlib import Path


def scene_key(payload, crs):
    files=[]
    for field in ('centerlines','surfaces','width_segments','valid_observation','road_probability'):
        path=Path(payload[field]).expanduser().resolve()
        family=([path.with_suffix(s) for s in ('.shp','.shx','.dbf','.prj','.cpg','.qix')]
                if path.suffix.lower()=='.shp' else [path,Path(str(path)+'.msk'),Path(str(path)+'.aux.xml'),Path(str(path)+'.ovr')])
        for item in family:
            stat=item.stat() if item.exists() else None
            files.append((str(item),None if stat is None else (stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino)))
    return str(crs),tuple(files)


def close_scene(scene):
    scene.probability.close()
    scene.width.cache_clear()
    scene.surface.cache_clear()


class SceneCache:
    def __init__(self):
        self.items=OrderedDict()

    def get(self, payload, crs, factory):
        key=scene_key(payload,crs)
        if key in self.items:
            self.items.move_to_end(key)
            print('[Fast batch timing] auto_scene_cache_hit=1',flush=True)
            return self.items[key]
        # Never retain a stale version or more than two period raster contexts.
        for old in list(self.items):
            if old[1][0][0]==key[1][0][0]:
                close_scene(self.items.pop(old))
        if len(self.items)>=2:
            _,old=self.items.popitem(last=False);close_scene(old)
        scene=factory()
        if scene_key(payload,crs)!=key:
            close_scene(scene)
            raise RuntimeError('Road scene inputs changed while being read; retry the change pair.')
        self.items[key]=scene
        print('[Fast batch timing] auto_scene_cache_miss=1',flush=True)
        return scene

    def close(self):
        for scene in self.items.values():close_scene(scene)
        self.items.clear()
