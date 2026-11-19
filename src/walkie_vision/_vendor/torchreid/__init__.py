"""OSNet x1.0, vendored from deep-person-reid (Torchreid, MIT).

osnet.py is the network definition from torchreid at commit f8cd150, with
only osnet_x1_0 kept. walkie_vision.reid wraps it: load_osnet() builds the
network and loads Kaiyang Zhou's MSMT17 re-ID weights; nothing else imports
it.
"""
