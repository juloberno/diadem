# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import tempfile
import os
import shutil
import uuid
from diadem.experiment.evaluation.export.export_image import exportImage
import subprocess

class VideoRenderer:
	def __init__(self, fig_path=None, video_name=None, clear_figures=True):
		video_dir_name = "vid_render_frames_{}".format(uuid.uuid4()) if video_name is None else video_name
		
		self.fig_path = fig_path if fig_path is not None else tempfile.gettempdir()

		self.video_frame_dir = os.path.join(self.fig_path, video_dir_name)
		if clear_figures and os.path.exists(self.video_frame_dir):
			print("video path already exists, clear existing one")
			shutil.rmtree(os.path.abspath(self.video_frame_dir))
		
		os.makedirs(self.video_frame_dir, exist_ok=True)

		self.frame_count = 0

	def catch_frame(self, fig, **kwargs):
		exportImage(os.path.join(self.video_frame_dir, "{}".format(self.frame_count)),fig=fig,type="png",**kwargs)
		self.frame_count += 1

	def export_video(self, filename, framerate, remove_video_dir=True):
		if os.path.isfile(filename):
			os.remove(filename)
		if not os.path.exists(os.path.dirname(filename)):
			os.makedirs(os.path.dirname(filename))
		cmd = "ffmpeg -y -framerate {} -i \'{}/%01d.png\' -vcodec h264 -force_key_frames 'expr:gte(t,n_forced*{})' -acodec pcm_s16le -s 1920x1080 -r 30 -b:v 36M -pix_fmt yuv420p -f mp4 \'{}.mp4\'".format(
			framerate, self.video_frame_dir, 1 / framerate, os.path.abspath(filename))
		os.system(cmd)

		if remove_video_dir:
			shutil.rmtree(os.path.abspath(self.video_frame_dir))
