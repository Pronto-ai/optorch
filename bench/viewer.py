import queue
import multiprocessing

import pangolin as pango
import OpenGL.GL as gl
import numpy as np

def start_viewer():
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=run_viewer, args=(q,))
    p.daemon = True
    p.start()
    return q

def run_viewer(q):
    w, h = 1024, 768
    f = 2000
    pango.CreateWindowAndBind('g2o', w, h)
    gl.glEnable(gl.GL_DEPTH_TEST)
    cam = pango.OpenGlRenderState(
        pango.ProjectionMatrix(w, h, f, f, w//2, h//2, 0.1, 100000),
        pango.ModelViewLookAt(
            1000., 1000., 1000.,
            0., 0., 0.,
            0., -1., 0.,
        )
    )
    handler = pango.Handler3D(cam)
    dcam = pango.CreateDisplay()
    dcam.SetBounds(0., 1., 0., 1., -w/h)
    dcam.SetHandler(handler)
    dcam.Activate()

    # nodes = [x.estimate().matrix() for x in optimizer.vertices().values()]
    # nodes = np.array(nodes)

    edges = None

    while not pango.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.15, 0.15, 0.15, 0.0)
        dcam.Activate(cam)

        try:
            edges = q.get(block=False)
        except queue.Empty:
            pass

        if edges is not None:
            gl.glLineWidth(1)
            gl.glColor3f(0.2, 1.0, 0.2)
            pango.DrawLines(edges[:,0], edges[:,1])

        pango.FinishFrame()
