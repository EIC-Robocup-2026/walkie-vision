"""Code copied from upstream projects, kept close to how upstream wrote it.

Each package is named after its upstream project and holds only the files the
vision node needs. Each file's header names the upstream URL, the commit, the
licence (MIT for both) and the changes made here; keep the licence notice and
the change list up to date when a file changes. Vendored code imports nothing
else from walkie_vision, and each package is imported only by the provider
that wraps it:

- insightface: SCRFD, ArcFace and the face alignment from InsightFace,
  wrapped by walkie_vision.providers.face_recognition.buffalo_l.
- torchreid: OSNet x1.0 from deep-person-reid (Torchreid), wrapped by
  walkie_vision.providers.reid.osnet.
"""
