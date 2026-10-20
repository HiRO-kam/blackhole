"""Numerical and non-window rendering checks: python validate.py"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from imaging import render_image, trajectory
from ray_engine import trace_batch


def main():
    b=np.array([0,4.5,5.19,5.2,7,12.])
    _,status,error=trace_batch(b,r0=30,sky=120)
    assert list(status)==[1,1,1,2,2,2]
    fine,st2,err2=trace_batch(b,r0=30,sky=120,scale=.5)
    assert np.array_equal(status,st2)
    image,reference,meta=render_image(256)
    assert meta['unresolved']==0
    assert np.all(image[127:129,127:129]==0)
    assert np.array_equal(image==0,(image==0)[::-1])
    for bi in b:
        path,st=trajectory(bi)
        assert st==('捕獲' if bi<np.sqrt(27) else '脱出')
        assert abs(np.linalg.norm(path[-1])-(2 if st=='捕獲' else 30))<1e-9
    # Compare numerical capture edge to finite-observer angular prediction.
    row=np.all(image[128]==0,axis=-1)
    measured=np.count_nonzero(row)/2
    alpha=np.arcsin(np.sqrt(27)*np.sqrt(1-2/30)/30)
    expected=128*np.tan(alpha)/np.tan(np.deg2rad(45/2))
    assert abs(measured-expected)<1
    out=Path(__file__).parent
    Image.fromarray(image).save(out/'example_image.jpg',quality=95,subsampling=0)
    assert Image.open(out/'example_image.jpg').size==(256,256)
    report={**meta,'conserved_quantity_max_error':float(error.max()),'shadow_radius_pixels':measured,'theory_radius_pixels':float(expected),'desktop_window_tested':False}
    (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report)

if __name__=='__main__': main()
