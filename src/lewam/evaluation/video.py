from fractions import Fraction

import av


def write_video(path, frames, fps=10):
    if not frames:
        return
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps))
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        for image in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
