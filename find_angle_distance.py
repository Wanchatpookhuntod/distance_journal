import math
class Find_angle_distance:

	def __init__(self):

		self.__adjecent_side = 193.4 # Distance In 193.4 Cm

		self.__horizontal_px = 320 # Px Of Hight View
		self.__horizontal_cm = 59.8316 # Cm Of Hight View

		self.__vertical_px = 240 # Px Of Width View
		self.__vertical_cm = 46.9785 # Cm Of Width View
		self.__px_plus = 13.33  # 240/18"
		self.__cm_plus = 2.54  # Inch : Cm

	def get_eye(self, eye_center, re_screen = 0):
		self.point_x = eye_center[0]  # Yxaxe Of Center Eyes
		self.point_y = eye_center[1]  # Y Axe Of Center Eyes
		self.re_screen = re_screen  # Set Screen Center

	def set_camera(self):
		# Px Of Change Screen Center
		# Cm Of Change Screen Center

		self.vertical_px_new = self.__vertical_px + \
											(self.__px_plus * self.re_screen)
		self.vertical_cm_new = self.__vertical_cm + \
											(self.__cm_plus * self.re_screen)

		return self.vertical_px_new, self.vertical_cm_new

	def change_point_start_vertical(self):
		# Check Status Up Or Down
		# Make Index Gaze View
		# New Point Start 0 Center

		if self.point_y < self.set_camera()[0] :
			self.status_gaze_vt = "OVER"
		else:
			self.status_gaze_vt = "UNDER"
		self.new_point_y = abs(self.point_y - self.set_camera()[0])

		return self.new_point_y , self.status_gaze_vt

	def estimate_angle_vertical(self):
		# Scale Cm : Px
		# Hight Cm Eye Point
		# Calculator Angle By Arctan
		self.H = self.set_camera()[0]  / self.set_camera()[1]
		self.eye_hight_cm = self.change_point_start_vertical()[0] / self.H
		self.angle_hight = (math.atan2(self.eye_hight_cm,
								self.__adjecent_side) * 180 / math.pi)

		return self.angle_hight, self.eye_hight_cm, self.H

	def change_point_start_horizontal(self):
		# Check Status Right Or Left
		# Make Index Gaze View
		# New Point Start 0 Center
		if self.point_x < self.__horizontal_px:
			self.status_gaze_hz = "RIGHT"
		else:
			self.status_gaze_hz = "LEFT"
		self.new_point_x = abs(self.point_x - self.__horizontal_px)

		return self.new_point_x, self.status_gaze_hz

	def estimate_angle_horizontal(self):
		# Scale Cm : Px
		# Width Cm Eye Point
		# Calculator Angle By Arctan
		self.W = self.__horizontal_px / self.__horizontal_cm
		self.eye_width_cm = self.change_point_start_horizontal()[0] / self.W
		self.angle_width = (math.atan2(self.eye_width_cm,
								self.__adjecent_side) * 180 / math.pi)

		return self.angle_width, self.eye_width_cm, self.W

	def estimate_distance(self, right_eye_x, est=2.5):
		# Equation Angle Tan
		# Compute Angle B
		# Compute Line AD to Cm
		# Compute Distance
		tan_round = lambda a : math.tan(math.radians(a))
		angle_B = math.atan2(abs(self.point_x - right_eye_x),
				abs(self.point_y - self.set_camera()[0]))*180/math.pi
		line_AD = tan_round(abs(90 - angle_B))*est
		estimate_distance_new = tan_round(abs\
				(90 - self.estimate_angle_vertical()[0]))*line_AD

		return estimate_distance_new

	# class Roi_eye_distance:
	# 	def roi(self, p36,p37,p39,p40,p):
