#!/usr/bin/env python

from manimlib.imports import *

# To watch one of these scenes, run the following:
# python -m manim example_scenes.py SquareToCircle -pl
#
# Use the flat -l for a faster rendering at a lower
# quality.
# Use -s to skip to the end and just save the final frame
# Use the -p to have the animation (or image, if -s was
# used) pop up once done.
# Use -n <number> to skip ahead to the n'th animation of a scene.
# Use -r <number> to specify a resolution (for example, -r 1080
# for a 1920x1080 video)

def FuncZero(p):
	x, y = p[:2]
	result = RIGHT
	return result

def FuncZeroVer(p):
	x, y = p[:2]
	result = UP
	return result

def FuncLinear(p):
	x, y = p[:2]
	result = x*UP/2
	return result

def FuncLinearVer(p):
	x, y = p[:2]
	result = y*RIGHT/2
	return result

def FuncSigmoid(p):
	x, y = p[:2]
	result = 3 * sigmoid(.75*p[0]) * RIGHT
	return result

def FuncSigmoidShift(p):
	x, y = p[:2]
	result = 3 * sigmoid(.75*p[0]-4) * RIGHT
	return result

def FuncSigmoidVer(p):
	x, y = p[:2]
	result = 3 * sigmoid(.75*p[1]) * UP
	return result

def FuncDiv(p):
	x, y = p[:2]
	result = x * RIGHT/2 + y * UP/2
	return result

def FuncDivShift(p):
	x, y = p[:2]
	result = x * RIGHT/2 + y * UP/2 + LEFT*2
	return result

def FuncCurl(p):
	x, y = p[:2]
	result = -y * RIGHT/2 + x * UP/2
	return result

def FuncDeform(p):
	x, y = p[:2]
	result = y * RIGHT
	return result

def four_swirls_function(p):
	x, y = p[:2]
	result = (y**3 - 4 * y) * RIGHT + (x**3 - 16 * x) * UP
	result *= 0.05
	norm = get_norm(result)
	if norm == 0:
		return result
	# result *= 2 * sigmoid(norm) / norm
	return result



class Intro(Scene):
	def construct(self):

		
		text = TextMobject(
			"Gradient Tensor",
			", ", 
			"Strain Rate Tensor",
			", and ",
			"Rotation Tensor")
		text2 = TextMobject("How to calculate") 
		text3 = TextMobject("translation, stretch, deformation, and rotation")
		text4 = TextMobject("and the relation with ")
		text5 = TextMobject("the ",
			"Stress Tensor",
			", ",
			"Divergence",
			", and ",
			"Curl")

		text[0].set_color(RED)
		text[2].set_color(YELLOW)
		text[4].set_color(BLUE)
		text5[1].set_color(TEAL)
		text5[3].set_color(ORANGE)
		text5[5].set_color(PURPLE)


		text.shift(2*UP)
		text3.next_to(text2,DOWN)
		text4.next_to(text3,DOWN*2)
		text5.next_to(text4,DOWN)
		text.scale(1.1)
		text2.scale(0.8)
		text3.scale(1)
		text4.scale(.8)
		text5.scale(1)

		self.play(Write(text))
		self.wait()
		self.play(Write(text2))
		self.play(Write(text3))
		self.wait()
		self.play(Write(text4))
		self.play(Write(text5))

		self.wait(2)


class FluidElementField(MovingCameraScene):
	CONFIG={
		"dot_kwargs":{
			"radius":.05,
			"color":WHITE,
			"fill_opacity": 1,
		}
	}
	def get_vector_label(self, vector, label, at_tip=False,	direction="left", rotate=False, color=None, label_scale_factor=VECTOR_LABEL_SCALE_FACTOR):
		if not isinstance(label, TexMobject):
			if len(label) == 1:
				label = "\\vec{\\textbf{%s}}" % label
			label = TexMobject(label)
			if color is None:
				color = vector.get_color()
			label.set_color(color)
		label.scale(label_scale_factor)
		label.add_background_rectangle()

		if at_tip:
			vect = vector.get_vector()
			vect /= get_norm(vect)
			label.next_to(vector.get_end(), vect, buff=SMALL_BUFF)
		else:
			angle = vector.get_angle()
			if not rotate:
				label.rotate(-angle, about_point=ORIGIN)
			if direction == "left":
				label.shift(-label.get_bottom() + 0.1 * UP)
			else:
				label.shift(-label.get_top() + 0.1 * DOWN)
			label.rotate(angle, about_point=ORIGIN)
			label.shift((vector.get_end() - vector.get_start()) / 2)
		return label

	def label_vector(self, vector, label, animate=True, **kwargs):
		label = self.get_vector_label(vector, label, **kwargs)
		if animate:
			self.play(Write(label, run_time=1))
		self.add(label)
		return label

	def FieldCreation(self):
		FieldZero = VectorField(four_swirls_function, 
			delta_x = .5, 
			delta_y = .5,
			x_max = int(np.ceil(FRAME_WIDTH / 2 + 4)),
			y_max = int(np.ceil(FRAME_HEIGHT / 2 + 4)),
			length_func = lambda norm: 0.9 * sigmoid(norm))

		dots=VGroup()
		for vector in FieldZero:
			dot = Dot(**self.dot_kwargs)
			dot.move_to(vector.get_start())
			dot.target = vector
			dots.add(dot)
		self.play(
			ShowCreation(dots),
			#run_time=2,
			)
		self.wait(2)

		self.play(ShowCreation(FieldZero))
		self.wait(2)

		move_submobjects_along_vector_field(dots, four_swirls_function)
		self.wait(5)
		self.remove(dots)
		self.wait()

		return FieldZero

	def FluidElement(self, FieldZero):
		v1 = Vector([0.5,3])
		v2 = Vector([4.5,3])

		rect = Rectangle(width=4,height=2)
		rect.move_to(RIGHT*2.5+UP*3)
		matrix = [[np.cos(-1 * TAU/16), np.cos(TAU/4-1 * TAU/16)], [np.sin(-1 * TAU / 16), np.sin(TAU/4-1 * TAU/16)]]
		matrix2 = [[1.2 * np.cos(1 * TAU/20),np.cos(TAU/4+1 * TAU/20)],[np.sin(1 * TAU / 20), -.3 + np.sin(TAU/4+1 * TAU/20)]]
		matrix2r = np.linalg.inv(matrix2)

		rect.apply_matrix(matrix)
		v1.apply_matrix(matrix)
		v2.apply_matrix(matrix)

		l3 = TexMobject("\\vec{\\textbf{v}}(\\vec{\\textbf{x}})")
		l3.next_to(v1.get_end())
		l3.scale(0.8)
		l3.add_background_rectangle()

		l4 = TexMobject("\\vec{\\textbf{v}}(\\vec{\\textbf{x}}+d\\vec{\\textbf{x}})")
		l4.move_to(v2.get_end()+UP*1.3+RIGHT*.7)
		l4.scale(0.8)
		l4.add_background_rectangle()

		dotLeft = Dot(v1.get_end()).fade(1)
		dotRight = Dot(v2.get_end()).fade(1)
		vLeft = FieldZero.get_vector(dotLeft.get_center())
		vRight = FieldZero.get_vector(dotRight.get_center())

		vLeft.add_updater(
			lambda mob: mob.become(FieldZero.get_vector(dotLeft.get_center()))
		)

		arrowDif = Vector(v2.get_end() - v1.get_end())
		arrowDif.shift(v1.get_end())
		arrowDif.set_color(RED)

		arrowDifx = Vector([v2.get_end()[0] - v1.get_end()[0],0])
		arrowDifx.shift(v1.get_end())
		arrowDifx.set_color(BLUE)

		arrowDify = Vector([0,v2.get_end()[1] - v1.get_end()[1]])
		arrowDify.shift(arrowDifx.get_end())
		arrowDify.set_color(BLUE)


		self.play(
		self.camera_frame.scale, 3/5,
		self.camera_frame.move_to, RIGHT*3.5+UP*2,
			run_time=3
			)
		self.wait()
		self.play(ShowCreation(v1))
		l1 = self.label_vector(v1,"x")
		self.wait()
		self.play(ShowCreation(v2))
		l2 = self.label_vector(v2,"\\vec{\\textbf{x}}+d\\vec{\\textbf{x}}")
		self.wait(3)

		self.remove(FieldZero)
		self.wait()

		self.add(dotLeft,vLeft)
		self.play(Write(l3))
		self.wait()
		self.add(dotRight,vRight)
		self.play(Write(l4))
		self.wait(2)

		self.play(ShowCreation(arrowDif))
		self.wait()

		self.play(
			dotLeft.move_to,v2.get_end(),
			run_time=4
		)
		dotLeft.move_to(v1.get_end())
		self.remove(arrowDif)
		self.wait()
		
		self.play(ShowCreation((rect)))
		self.wait(3)

		self.remove(v1, l1, v2, l2, l3, l4, vLeft, vRight)
		self.add(FieldZero)

		self.play(ApplyMethod(rect.apply_matrix, matrix2))
		self.wait(2)

		self.remove(FieldZero)
		rect.apply_matrix(matrix2r)
		self.add(v1, l1, v2, l2, l3, l4, vLeft, vRight)



		return v1, v2, l3, l4, dotLeft, dotRight, arrowDifx, arrowDify

	def Equation(self, plane, v1, v2, l3, l4, dotLeft, dotRight, arrowDifx, arrowDify):
		self.play(
			self.camera_frame.scale, 5/3,
			self.camera_frame.move_to, RIGHT*3,
			run_time=3
			)
		self.remove(plane)
		self.wait()

		text1 = TexMobject("\\textbf{v}(\\textbf{x}+d\\textbf{x}) ",
			"= \\textbf{v}(\\textbf{x}) + ", 
			"\\pdv{\\textbf{v}(\\textbf{x})}{x}",
			"dx + ",
			"\\pdv{\\textbf{v}(\\textbf{x})}{y}", 
			"dy + ",
			"\\pdv{\\textbf{v}(\\textbf{x})}{z}",
			"dz")

		text1[2].set_color(RED)
		text1[4].set_color(RED)
		text1[6].set_color(RED)

		textDiv = TexMobject("d\\textbf{x} \\cdot",
		 "\\nabla \\textbf{v}(\\textbf{x}) ")

		textDiv[1].set_color(RED)

		textV = TexMobject("\\textbf{v}(\\textbf{x}) = u \\hat{\\textbf{x}} + v \\hat{\\textbf{y}} + w \\hat{\\textbf{z}}")

		text1.shift(2*DOWN)
		text1.shift(RIGHT*3)
		textDiv.next_to(text1[1], RIGHT)
		textV.next_to(text1, DOWN)
		

		self.play(ReplacementTransform(l4.copy(),text1[0]))
		self.wait(2)
		self.play(ReplacementTransform(l3.copy(),text1[1]))
		self.wait(2)
		# self.play(ShowCreation(arrowDif))


		self.play(
			dotLeft.shift,[v2.get_end()[0] - v1.get_end()[0],0,0],
			ShowCreation(arrowDifx),
			run_time=3
		)
		self.wait()

		
		self.play(
			dotLeft.move_to,v2.get_end(),
			ShowCreation(arrowDify),
			run_time=3
		)
		self.wait()

		self.play(
			ReplacementTransform(arrowDifx,text1[2:4]),
			)
		dotLeft.move_to(v1.get_end())
		self.wait(1)

		self.play(
			ReplacementTransform(arrowDify,text1[4:6]),
			)
		self.wait(1)

		self.play(Write(text1[6:]))
		self.wait(3)
		self.play(ReplacementTransform(text1[2:],textDiv))
		self.wait(3)
		self.play(Write(textV))
		self.wait(3)


	def construct(self):
		plane = self.plane = NumberPlane(x_max = 2*FRAME_X_RADIUS, y_max = 2*FRAME_Y_RADIUS)
		plane.add_coordinates()
		plane.remove(plane.coordinate_labels[-1])
		self.play(ShowCreation(plane))
		self.wait()



		FieldZero = self.FieldCreation()

		v1, v2, l3, l4, dotLeft, dotRight, arrowDifx, arrowDify = self.FluidElement(FieldZero)

		self.Equation(plane, v1, v2, l3, l4, dotLeft, dotRight, arrowDifx, arrowDify)


class VelocityGradientTensor(Scene):
	def construct(self):
		matrix = Matrix(
			[["\\pdv{u}{x}", "\\pdv{v}{x}", "\\pdv{w}{x}"], 
			["\\pdv{u}{y}", "\\pdv{v}{y}", "\\pdv{w}{y}"],
			["\\pdv{u}{z}", "\\pdv{v}{z}", "\\pdv{w}{z}"]
			], v_buff=1.5)

		text1 = TextMobject("Velocity gradient tensor")
		textL = TexMobject("\\underline{\\underline{L}}",
			" \\equiv ")
		textDiv = TexMobject("(",
			"\\nabla \\textbf{v}",
			")^T =")
		textLT = TexMobject("(\\underline{\\underline{L}})^T \\equiv ")
		textDivT = TexMobject("",
			"\\nabla \\textbf{v} ",
			"=")
		textJ = TexMobject("=(\\nabla \\underline{\\underline{J}})^T")
		
		textJT = TexMobject("= \\underline{\\underline{J}}")
		
		matrixT = Matrix(
			[["\\pdv{u}{x}", "\\pdv{u}{y}", "\\pdv{u}{z}"], 
			["\\pdv{v}{x}", "\\pdv{v}{y}", "\\pdv{v}{z}"],
			["\\pdv{w}{x}", "\\pdv{w}{y}", "\\pdv{w}{z}"]
			], v_buff=1.5)

		text1.set_color(RED)
		textL[0].set_color(RED)
		textDiv[1].set_color(RED)
		textDivT[0].set_color(RED)
		matrix[1].set_color(RED)
		matrix[2].set_color(RED)
		matrixT[1].set_color(RED)
		matrixT[2].set_color(RED)


		text1.to_edge(UP)
		textDiv.next_to(matrix, LEFT)
		textDivT.next_to(matrixT, LEFT)
		textL.next_to(textDiv, LEFT)
		textLT.next_to(textDivT, LEFT)
		textJ.next_to(matrix, RIGHT)
		textJT.next_to(matrixT, RIGHT)

		textM1 = TexMobject("\\pdv{u}{x}")

		
		self.play(Write(textDivT))
		self.play(Write(matrixT))
		self.wait(5)

		self.play(Write(textJT))
		self.wait(2)

		self.play(ReplacementTransform(textDivT,textDiv), 
			ReplacementTransform(matrixT[0][1],matrix[0][3]),
			ReplacementTransform(matrixT[0][2],matrix[0][6]),
			ReplacementTransform(matrixT[0][5],matrix[0][7]),
			ReplacementTransform(matrixT[0][7],matrix[0][5]),
			ReplacementTransform(matrixT[0][6],matrix[0][2]),
			ReplacementTransform(matrixT[0][3],matrix[0][1]), 
			ReplacementTransform(textJT,textJ))
		self.wait()
		self.play(Write(textL))
		self.wait()

		self.play(Write(text1))
		self.wait(2)

		self.remove(matrix[0][3],matrix[0][6],matrix[0][7],matrix[0][5],matrix[0][2],matrix[0][1])

		self.play(FadeOut(text1), FadeOut(textDiv), FadeOut(textL), FadeOut(matrixT), FadeOut(textJ), ReplacementTransform(matrix[0][0], textM1))
		self.wait(3)
		self.play(ApplyMethod(textM1.to_corner, UL))


class FieldDuDx(Scene):
	def add_plane(self):
		plane = self.plane = NumberPlane()
		plane.add_coordinates()
		plane.remove(plane.coordinate_labels[-1])
		self.add(plane)


	def construct(self):
		AxisArrow = Vector(RIGHT)
		self.add_plane()
		FieldSigmoid = VectorField(FuncSigmoid)
		FieldZero = VectorField(FuncZero)
		FieldSigmoidVer = VectorField(FuncSigmoidVer)

		textM1 = TexMobject("\\pdv{u}{x}")
		textM1.to_corner(UL)
		textM1.add_background_rectangle()


		self.add_foreground_mobject(textM1)
		self.add(FieldZero)
		self.add(AxisArrow)
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*1.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*2.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*3.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*4.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*5.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*6.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*7.5))
		self.wait(.25)
		self.remove(AxisArrow)
		AxisArrow.move_to(RIGHT*.5)

		self.play(ReplacementTransform(FieldZero,FieldSigmoid))
		self.wait(1)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*1.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*2.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*3.5))
		self.wait(.25)
		self.remove(AxisArrow)
		AxisArrow.move_to(RIGHT*.5)

		self.play(ReplacementTransform(FieldSigmoid,FieldSigmoidVer))
		self.wait(1)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*1.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*2.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*3.5))
		self.wait()


class VelocityGradientTensorDvDx(Scene):
	def construct(self):
		matrix = Matrix(
			[["\\pdv{u}{x}", "\\pdv{v}{x}", "\\pdv{w}{x}"], 
			["\\pdv{u}{y}", "\\pdv{v}{y}", "\\pdv{w}{y}"],
			["\\pdv{u}{z}", "\\pdv{v}{z}", "\\pdv{w}{z}"]
			], v_buff=1.5)

		text1 = TextMobject("Velocity gradient tensor")
		textL = TexMobject("\\underline{\\underline{L}} ",
			"\\equiv (",
			"\\nabla \\textbf{v}",
			")^T =")
		textJ = TexMobject("=( \\underline{\\underline{J}})^T")
		
		textL[0].set_color(RED)
		textL[2].set_color(RED)
		text1.set_color(RED)
		matrix[1].set_color(RED)
		matrix[2].set_color(RED)

		text1.to_edge(UP)
		textL.next_to(matrix, LEFT)
		textJ.next_to(matrix, RIGHT)

		textM2 = TexMobject("\\pdv{v}{x}")

		self.add(text1)
		self.add(textL)
		self.add(matrix)
		self.add(textJ)
		self.wait()

		self.play(FadeOut(text1), FadeOut(textL), FadeOut(matrix), FadeOut(textJ), ReplacementTransform(matrix[0][1], textM2))
		self.wait(3)
		self.play(ApplyMethod(textM2.to_corner, UL))


class FieldDvDx(Scene):
	def add_plane(self):
		plane = self.plane = NumberPlane()
		plane.add_coordinates()
		plane.remove(plane.coordinate_labels[-1])
		self.add(plane)


	def construct(self):
		AxisArrow = Vector(RIGHT)
		self.add_plane()
		FieldSigmoid = VectorField(FuncSigmoid)
		FieldZero = VectorField(FuncZero)
		FieldSigmoidVer = VectorField(FuncSigmoidVer)
		FieldLinear = VectorField(FuncLinear)

		textM1 = TexMobject("\\pdv{v}{x}")
		textM1.to_corner(UL)
		textM1.add_background_rectangle()


		self.add_foreground_mobject(textM1)
		self.add(FieldSigmoid)
		self.add(AxisArrow)
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*1.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*2.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*3.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*4.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*5.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*6.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*7.5))
		self.wait(.25)
		self.remove(AxisArrow)
		AxisArrow.move_to(RIGHT*.5)

		self.play(ReplacementTransform(FieldSigmoid,FieldSigmoidVer))
		self.wait(1)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*1.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*2.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*3.5))
		self.wait(.25)
		self.remove(AxisArrow)
		AxisArrow.move_to(RIGHT*.5)

		self.play(ReplacementTransform(FieldSigmoidVer,FieldLinear))
		self.wait(1)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*1.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*2.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, RIGHT*3.5))
		self.wait()


class VelocityGradientTensorDvDy(Scene):
	def construct(self):
		matrix = Matrix(
			[["\\pdv{u}{x}", "\\pdv{v}{x}", "\\pdv{w}{x}"], 
			["\\pdv{u}{y}", "\\pdv{v}{y}", "\\pdv{w}{y}"],
			["\\pdv{u}{z}", "\\pdv{v}{z}", "\\pdv{w}{z}"]
			], v_buff=1.5)

		text1 = TextMobject("Velocity gradient tensor")
		textL = TexMobject("\\underline{\\underline{L}} ",
			"\\equiv (",
			"\\nabla \\textbf{v}",
			")^T =")
		textJ = TexMobject("=( \\underline{\\underline{J}})^T")
		
		textL[0].set_color(RED)
		textL[2].set_color(RED)
		text1.set_color(RED)
		matrix[1].set_color(RED)
		matrix[2].set_color(RED)

		text1.to_edge(UP)
		textL.next_to(matrix, LEFT)
		textJ.next_to(matrix, RIGHT)

		textM3 = TexMobject("\\pdv{v}{y}")

		self.add(text1)
		self.add(textL)
		self.add(matrix)
		self.add(textJ)
		self.wait()

		self.play(FadeOut(text1), FadeOut(textL), FadeOut(matrix), FadeOut(textJ), ReplacementTransform(matrix[0][4], textM3))
		self.wait(3)
		self.play(ApplyMethod(textM3.to_corner, UL))


class FieldDvDy(Scene):
	def add_plane(self):
		plane = self.plane = NumberPlane()
		plane.add_coordinates()
		plane.remove(plane.coordinate_labels[-1])
		self.add(plane)


	def construct(self):
		AxisArrow = Vector(UP)
		self.add_plane()
		FieldSigmoid = VectorField(FuncSigmoid)
		FieldZero = VectorField(FuncZero)
		FieldSigmoidVer = VectorField(FuncSigmoidVer)
		FieldZeroVer = VectorField(FuncZeroVer)

		textM3 = TexMobject("\\pdv{v}{y}")
		textM3.to_corner(UL)
		textM3.add_background_rectangle()

		self.add_foreground_mobject(textM3)
		self.add(FieldZero)
		self.add(AxisArrow)
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*1.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*2.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*3.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*4.5))
		self.wait(.25)
		self.remove(AxisArrow)
		AxisArrow.move_to(UP*.5)

		self.play(ReplacementTransform(FieldZero,FieldZeroVer))
		self.wait(1)
		self.add(AxisArrow)
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*1.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*2.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*3.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*4.5))
		self.wait(.25)
		self.remove(AxisArrow)
		AxisArrow.move_to(UP*.5)

		self.play(ReplacementTransform(FieldZeroVer,FieldSigmoidVer))
		self.wait(1)
		self.add(AxisArrow)
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*1.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*2.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*3.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*4.5))
		self.wait()


class VelocityGradientTensorDuDy(Scene):
	def construct(self):
		matrix = Matrix(
			[["\\pdv{u}{x}", "\\pdv{v}{x}", "\\pdv{w}{x}"], 
			["\\pdv{u}{y}", "\\pdv{v}{y}", "\\pdv{w}{y}"],
			["\\pdv{u}{z}", "\\pdv{v}{z}", "\\pdv{w}{z}"]
			], v_buff=1.5)

		text1 = TextMobject("Velocity gradient tensor")
		textL = TexMobject("\\underline{\\underline{L}} ",
			"\\equiv (",
			"\\nabla \\textbf{v}",
			")^T =")
		textJ = TexMobject("=( \\underline{\\underline{J}})^T")
		
		textL[0].set_color(RED)
		textL[2].set_color(RED)
		text1.set_color(RED)
		matrix[1].set_color(RED)
		matrix[2].set_color(RED)

		text1.to_edge(UP)
		textL.next_to(matrix, LEFT)
		textJ.next_to(matrix, RIGHT)

		textM3 = TexMobject("\\pdv{u}{y}")

		self.add(text1)
		self.add(textL)
		self.add(matrix)
		self.add(textJ)
		self.wait()

		self.play(FadeOut(text1), FadeOut(textL), FadeOut(matrix), FadeOut(textJ), ReplacementTransform(matrix[0][3], textM3))
		self.wait(3)
		self.play(ApplyMethod(textM3.to_corner, UL))


class FieldDuDy(Scene):
	def add_plane(self):
		plane = self.plane = NumberPlane()
		plane.add_coordinates()
		plane.remove(plane.coordinate_labels[-1])
		self.add(plane)


	def construct(self):
		AxisArrow = Vector(UP)
		self.add_plane()
		FieldSigmoid = VectorField(FuncSigmoid)
		FieldZero = VectorField(FuncZero)
		FieldSigmoidVer = VectorField(FuncSigmoidVer)
		FieldZeroVer = VectorField(FuncZeroVer)
		FieldLinearVer = VectorField(FuncLinearVer)


		textM3 = TexMobject("\\pdv{u}{y}")
		textM3.to_corner(UL)
		textM3.add_background_rectangle()

		self.add_foreground_mobject(textM3)
		self.add(FieldZero)
		self.add(AxisArrow)
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*1.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*2.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*3.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*4.5))
		self.wait(.25)
		self.remove(AxisArrow)
		AxisArrow.move_to(UP*.5)

		self.play(ReplacementTransform(FieldZero,FieldLinearVer))
		self.wait(1)
		self.add(AxisArrow)
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*1.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*2.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*3.5))
		self.wait(.25)
		self.play(ApplyMethod(AxisArrow.move_to, UP*4.5))
		self.wait()


class Zero(Scene):
	def construct(self):
		text1 = TextMobject("If the ",
			"velocity gradient tensor ",
			"is zero:")
		text2 = TexMobject("\\underline{\\underline{L}} ",
			"\\equiv (",
			"\\nabla \\textbf{v}",
			")^T = \\underline{\\underline{0}} ")

		text3 = TexMobject("\\textbf{v}(\\textbf{x}+d\\textbf{x}) = \\textbf{v}(\\textbf{x}) + d\\textbf{x} \\cdot",
		 "\\nabla \\textbf{v}(\\textbf{x}) ")

		text3b = TexMobject("\\textbf{v}(\\textbf{x}+d\\textbf{x}) = \\textbf{v}(\\textbf{x}) + d\\textbf{x} \\cdot",
		 "\\underline{\\underline{0}}")

		text4 = TexMobject("\\textbf{v}(\\textbf{x}+d\\textbf{x}) = \\textbf{v}(\\textbf{x})")

		text5 = TextMobject("only translation:")
		text6 = TextMobject("all points move in the same direction")
		text7 = TextMobject("with the same speed")


		text1[1].set_color(RED)
		text2[0].set_color(RED)
		text2[2].set_color(RED)
		text3[1].set_color(RED)


		text1.next_to(text2,UP*4)
		text3.next_to(text2,DOWN*1)
		text3b.next_to(text2,DOWN*1)
		text4.next_to(text2,DOWN*1)
		text5.next_to(text4,DOWN*2)
		text6.next_to(text5,DOWN*1)
		text7.next_to(text6,DOWN*1)


		self.play(Write(text1))
		self.play(Write(text2))
		self.wait(2)
		self.play(Write(text3))
		self.wait(1)
		self.play(ReplacementTransform(text2, text3b[1]))
		self.play(ReplacementTransform(text3, text3b))
		self.wait(2)
		self.play(ReplacementTransform(text3b, text4))
		self.wait(2)
		self.play(Write(text5))
		self.play(Write(text6))
		self.play(Write(text7))

		self.wait(3)


class FieldTranslationScene(Scene):
	CONFIG={
		"dot_kwargs":{
			"radius":.05,
			"color":YELLOW,
			"fill_opacity": 1,
		}
	}



	def construct(self):
		FieldZero = VectorField(FuncZero)

		v1 = Vector([0.5,3])
		v2 = Vector([4.5,3])
		v5 = Vector([-6,0])
		v6 = Vector([-2,0])

		rect = Rectangle(width=4,height=2)
		rect.move_to(RIGHT*2.5+UP*3)
		matrix = [[np.cos(-1 * TAU/16), np.cos(TAU/4-1 * TAU/16)], [np.sin(-1 * TAU / 16), np.sin(TAU/4-1 * TAU/16)]]
		matrixInv = [[np.cos(1 * TAU/16), np.cos(TAU/4+1 * TAU/16)], [np.sin(1 * TAU / 16), np.sin(TAU/4+1 * TAU/16)]]
		
		rect.apply_matrix(matrix)
		v1.apply_matrix(matrix)
		v2.apply_matrix(matrix)

		title = TextMobject("Translation")
		title.add_background_rectangle()
		title.to_edge(UP)

		# self.add_foreground_mobject(text)

		# l3 = TexMobject("\\vec{\\textbf{v}}(\\vec{\\textbf{x}})")
		# l3.next_to(v1.get_end())
		# l3.scale(0.8)
		# l3.add_background_rectangle()

		# l4 = TexMobject("\\vec{\\textbf{v}}(\\vec{\\textbf{x}}+d\\vec{\\textbf{x}})")
		# l4.move_to(v2.get_end()+UP*1.3+RIGHT*.7)
		# l4.scale(0.8)
		# l4.add_background_rectangle()

		# dotLeft = Dot(v1.get_end()).fade(1)
		# dotRight = Dot(v2.get_end()).fade(1)
		# vLeft = FieldZero.get_vector(dotLeft.get_center())
		# vRight = FieldZero.get_vector(dotRight.get_center())

		# vLeft.add_updater(
		# 	lambda mob: mob.become(FieldZero.get_vector(dotLeft.get_center()))
		# )

		# corner1 = Dot([0,1,0], color=WHITE)
		# corner2 = Dot([0,-1,0], color=WHITE)
		# corner3 = Dot([-4,-1,0], color=WHITE)
		# corner4 = Dot([-4,1,0], color=WHITE)
		leftdot = Dot([-6,0,0], color=RED)
		rightdot = Dot([-2,0,0], color=BLUE)


		# corners=VGroup()
		# for vector in FieldZero:
		# 	dot = Dot(**self.dot_kwargs)
		# 	dot.move_to(vector.get_start())
		# 	dot.target = vector
		# 	corners.add(dot)

		self.add(rect, v1, v2)
		self.play(
			ApplyMethod(rect.apply_matrix, matrixInv),
			ApplyMethod(v1.apply_matrix, matrixInv),
			ApplyMethod(v2.apply_matrix, matrixInv)
			)
		self.play(
			ApplyMethod(rect.move_to, LEFT*4),
			ReplacementTransform(v1, v5),
			ReplacementTransform(v2, v6)
			)
		self.wait()
		
		self.add(FieldZero)
		self.wait()
		self.remove(v5, v6)
		self.wait()
		self.add(leftdot, rightdot)
		self.play(Write(title))
		self.wait()

		for dot in leftdot,rightdot:
			move_submobjects_along_vector_field(
				dot,
				lambda p: FuncZero(p)
			)
		move_submobjects_along_vector_field(rect, FuncZero)
		self.wait(8)

		for dot in leftdot,rightdot:
			dot.clear_updaters()
		rect.clear_updaters()
		self.wait()


class Split(Scene):
	def construct(self):
		

		text1 = TextMobject("The ",
			"velocity gradient tensor ",
			"can be split")
		text2 = TextMobject("into a ",
			"strain rate tensor ",
			"and a ",
			"rotation tensor")
		text3 = TexMobject("\\underline{\\underline{L}}", 
			"= (",
			"\\nabla \\textbf{v}",
			")^T = ",
			"\\underline{\\underline{D}} ",
			"+ ",
			"\\underline{\\underline{\\Omega}}")

		text1[1].set_color(RED)
		text2[1].set_color(YELLOW)
		text2[3].set_color(BLUE)

		text3[0].set_color(RED)
		text3[2].set_color(RED)
		text3[4].set_color(YELLOW)
		text3[6].set_color(BLUE)
		text1.next_to(text2,UP)
		text3.next_to(text2,DOWN*2)

		

		self.play(Write(text1))
		self.play(Write(text2))
		self.play(Write(text3[0:4]))
		self.play(Indicate(text3[4]))
		self.play(Write(text3[5]))
		self.play(Indicate(text3[6]))
		

		self.wait(2)


class StrainRateTensor(Scene):
	def construct(self):
		matrix = Matrix(
			[["\\pdv{u}{x}", "{1 \\over 2}(\\pdv{v}{x}+\\pdv{u}{y})", "{1 \\over 2}(\\pdv{w}{x}+\\pdv{u}{z})"], 
			["{1 \\over 2}(\\pdv{u}{y}+\\pdv{v}{x})", "\\pdv{v}{y}", "{1 \\over 2}(\\pdv{w}{y}+\\pdv{v}{z})"],
			["{1 \\over 2}(\\pdv{u}{z}+\\pdv{w}{x})", "{1 \\over 2}(\\pdv{v}{z}+\\pdv{w}{y})", "\\pdv{w}{z}"]
			], v_buff=1.5, h_buff=3.5)

		text1 = TextMobject("Strain rate tensor")
		text = TexMobject("\\underline{\\underline{D}}",
			" =")

		text1.set_color(YELLOW)
		text[0].set_color(YELLOW)
		matrix[1].set_color(YELLOW)
		matrix[2].set_color(YELLOW)

		text1.to_edge(UP)
		text.shift(LEFT*5)
		matrix.next_to(text)

		self.play(Write(text1))
		self.play(Write(text))
		self.play(Write(matrix))

		self.wait(6)


class StrainRateTensorStress(Scene):
	def construct(self):
		matrix = Matrix(
			[["\\pdv{u}{x}", "{1 \\over 2}(\\pdv{v}{x}+\\pdv{u}{y})", "{1 \\over 2}(\\pdv{w}{x}+\\pdv{u}{z})"], 
			["{1 \\over 2}(\\pdv{u}{y}+\\pdv{v}{x})", "\\pdv{v}{y}", "{1 \\over 2}(\\pdv{w}{y}+\\pdv{v}{z})"],
			["{1 \\over 2}(\\pdv{u}{z}+\\pdv{w}{x})", "{1 \\over 2}(\\pdv{v}{z}+\\pdv{w}{y})", "\\pdv{w}{z}"]
			], v_buff=1.5, h_buff=3.5)

		text1 = TextMobject("Constitutive relation:")
		text2 = TextMobject("the ",
			"strain rate tensor ",
			"and the ",
			"viscous stress tensor")
		D = TexMobject("\\underline{\\underline{D}}",
			" =")
		Dtau = TexMobject("2 \\mu",
			"\\underline{\\underline{D}}")
		tau = TexMobject("\\underline{\\underline{\\tau}} ", "= ")

		D[0].set_color(YELLOW)
		matrix[1].set_color(YELLOW)
		matrix[2].set_color(YELLOW)
		Dtau[1].set_color(YELLOW)
		text2[1].set_color(YELLOW)
		text2[3].set_color(TEAL)
		tau[0].set_color(TEAL)

		text1.to_edge(UP)
		text2.next_to(text1, DOWN)
		D.shift(LEFT*5)
		matrix.next_to(D)
		Dtau.move_to(RIGHT)
		tau.next_to(Dtau, LEFT)


		matrixTau = Matrix(
			[["2 \\pdv{u}{x}", "(\\pdv{v}{x}+\\pdv{u}{y})", "(\\pdv{w}{x}+\\pdv{u}{z})"], 
			["(\\pdv{u}{y}+\\pdv{v}{x})", "2 \\pdv{v}{y}", "(\\pdv{w}{y}+\\pdv{v}{z})"],
			["(\\pdv{u}{z}+\\pdv{w}{x})", "(\\pdv{v}{z}+\\pdv{w}{y})", "2 \\pdv{w}{z}"]
			], v_buff=1.5, h_buff=3.5)


		self.add(D)
		self.add(matrix)
		self.play(FadeOut(matrix), FadeOut(D[1]))
		self.play(Write(text2))
		self.play(Write(text1))
		self.play(ApplyMethod(D[0].move_to, RIGHT))
		self.play(ReplacementTransform(D[0],Dtau), Write(tau))


		self.wait(6)


class StressTensorThreeD(ThreeDScene):
	def construct(self):
		axes = ThreeDAxes()
		cube = Cube()
		prism = Prism(dimensions = [4,4,4])
		prism2 = Prism(dimensions = [2,2,2])
		axes.add(axes.get_axis_labels())

		vecxx = Vector([1,0,0])
		vecxy = Vector([0,-1,0])
		vecxz = Vector([0,0,1])
		vecyx = Vector([1,0,0])
		vecyy = Vector([0,-1,0])
		vecyz = Vector([0,0,1])
		veczx = Vector([1,0,0])
		veczy = Vector([0,-1,0])
		veczz = Vector([0,0,1])
		vecxx.shift(RIGHT*2)
		vecxy.shift(RIGHT*2)
		vecxz.shift(RIGHT*2)
		vecyx.shift(DOWN*2)
		vecyy.shift(DOWN*2)
		vecyz.shift(DOWN*2)
		veczx.shift(OUT*2)
		veczy.shift(OUT*2)
		veczz.shift(OUT*2)
		vecxx.set_color(RED)
		vecxy.set_color(YELLOW)
		vecxz.set_color(GREEN)
		vecyx.set_color(RED)
		vecyy.set_color(YELLOW)
		vecyz.set_color(GREEN)
		veczx.set_color(RED)
		veczy.set_color(YELLOW)
		veczz.set_color(GREEN)



		self.set_camera_orientation(phi=75 * DEGREES)
		self.begin_ambient_camera_rotation(rate=0.2)

		# self.add(axes)
		self.add(prism)
		self.add(vecxx,vecxy,vecxz,vecyx,vecyy,vecyz,veczx,veczy,veczz)
		self.wait(5)
		self.remove(vecxx,vecxy,vecxz,vecyx,vecyy,vecyz,veczx,veczy,veczz)
		self.play(ReplacementTransform(prism,prism2))


class StressTensorAll(ThreeDScene):
	def construct(self):
		axes = ThreeDAxes()
		prism1 = Prism(dimensions = [2,2,2])
		axes.add(axes.get_axis_labels())

		matrixxx = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
		matrixxy = [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
		matrixxz = [[1, 0, 0], [0, 1, 0], [1, 0, 1]]

		matrixyy = [[1, 0, 0], [0, 2, 0], [0, 0, 1]]
		matrixyx = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
		matrixyz = [[1, 0, 0], [0, 1, 0], [0, 1, 1]]

		matrixzz = [[1, 0, 0], [0, 1, 0], [0, 0, 2]]
		matrixzx = [[1, 0, 1], [0, 1, 0], [0, 0, 1]]
		matrixzy = [[1, 0, 0], [0, 1, 1], [0, 0, 1]]


		rmatrixxx = [[1/2, 0, 0], [0, 1, 0], [0, 0, 1]]
		rmatrixxy = [[1, 0, 0], [-1, 1, 0], [0, 0, 1]]
		rmatrixxz = [[1, 0, 0], [0, 1, 0], [-1, 0, 1]]

		rmatrixyy = [[1, 0, 0], [0, 1/2, 0], [0, 0, 1]]
		rmatrixyx = [[1, -1, 0], [0, 1, 0], [0, 0, 1]]
		rmatrixyz = [[1, 0, 0], [0, 1, 0], [0, -1, 1]]

		rmatrixzz = [[1, 0, 0], [0, 1, 0], [0, 0, 1/2]]
		rmatrixzx = [[1, 0, -1], [0, 1, 0], [0, 0, 1]]
		rmatrixzy = [[1, 0, 0], [0, 1, -1], [0, 0, 1]]

		
		vecxxp1 = Vector([1,0,0])
		vecxyp1 = Vector([0,1,0])
		vecxzp1 = Vector([0,0,1])
		vecyxp1 = Vector([1,0,0])
		vecyyp1 = Vector([0,1,0])
		vecyzp1 = Vector([0,0,1])
		veczxp1 = Vector([1,0,0])
		veczyp1 = Vector([0,1,0])
		veczzp1 = Vector([0,0,1])
		vecxxp1.shift(RIGHT)
		vecxyp1.shift(RIGHT)
		vecxzp1.shift(RIGHT)
		vecyxp1.shift(UP)
		vecyyp1.shift(UP)
		vecyzp1.shift(UP)
		veczxp1.shift(OUT)
		veczyp1.shift(OUT)
		veczzp1.shift(OUT)
		vecxxp1.set_color(RED)
		vecxyp1.set_color(YELLOW)
		vecxzp1.set_color(GREEN)
		vecyxp1.set_color(RED)
		vecyyp1.set_color(YELLOW)
		vecyzp1.set_color(GREEN)
		veczxp1.set_color(RED)
		veczyp1.set_color(YELLOW)
		veczzp1.set_color(GREEN)

		vecxxn1 = Vector([-1,0,0])
		vecxyn1 = Vector([0,-1,0])
		vecxzn1 = Vector([0,0,-1])
		vecyxn1 = Vector([-1,0,0])
		vecyyn1 = Vector([0,-1,0])
		vecyzn1 = Vector([0,0,-1])
		veczxn1 = Vector([-1,0,0])
		veczyn1 = Vector([0,-1,0])
		veczzn1 = Vector([0,0,-1])
		vecxxn1.shift(LEFT)
		vecxyn1.shift(LEFT)
		vecxzn1.shift(LEFT)
		vecyxn1.shift(DOWN)
		vecyyn1.shift(DOWN)
		vecyzn1.shift(DOWN)
		veczxn1.shift(IN)
		veczyn1.shift(IN)
		veczzn1.shift(IN)
		vecxxn1.set_color(RED)
		vecxyn1.set_color(YELLOW)
		vecxzn1.set_color(GREEN)
		vecyxn1.set_color(RED)
		vecyyn1.set_color(YELLOW)
		vecyzn1.set_color(GREEN)
		veczxn1.set_color(RED)
		veczyn1.set_color(YELLOW)
		veczzn1.set_color(GREEN)

		vecxxp2 = Vector([1,0,0])
		vecxyp2 = Vector([0,1,0])
		vecxzp2 = Vector([0,0,1])
		vecyxp2 = Vector([1,0,0])
		vecyyp2 = Vector([0,1,0])
		vecyzp2 = Vector([0,0,1])
		veczxp2 = Vector([1,0,0])
		veczyp2 = Vector([0,1,0])
		veczzp2 = Vector([0,0,1])
		vecxxp2.shift(RIGHT*2)
		vecxyp2.shift(RIGHT+UP)
		vecxzp2.shift(RIGHT+OUT)
		vecyxp2.shift(UP+RIGHT)
		vecyyp2.shift(UP*2)
		vecyzp2.shift(UP+OUT)
		veczxp2.shift(OUT+RIGHT)
		veczyp2.shift(OUT+UP)
		veczzp2.shift(OUT*2)
		vecxxp2.set_color(RED)
		vecxyp2.set_color(YELLOW)
		vecxzp2.set_color(GREEN)
		vecyxp2.set_color(RED)
		vecyyp2.set_color(YELLOW)
		vecyzp2.set_color(GREEN)
		veczxp2.set_color(RED)
		veczyp2.set_color(YELLOW)
		veczzp2.set_color(GREEN)

		vecxxn2 = Vector([-1,0,0])
		vecxyn2 = Vector([0,-1,0])
		vecxzn2 = Vector([0,0,-1])
		vecyxn2 = Vector([-1,0,0])
		vecyyn2 = Vector([0,-1,0])
		vecyzn2 = Vector([0,0,-1])
		veczxn2 = Vector([-1,0,0])
		veczyn2 = Vector([0,-1,0])
		veczzn2 = Vector([0,0,-1])
		vecxxn2.shift(LEFT*2)
		vecxyn2.shift(LEFT+DOWN)
		vecxzn2.shift(LEFT+IN)
		vecyxn2.shift(DOWN+LEFT)
		vecyyn2.shift(DOWN*2)
		vecyzn2.shift(DOWN+IN)
		veczxn2.shift(IN+LEFT)
		veczyn2.shift(IN+DOWN)
		veczzn2.shift(IN*2)
		vecxxn2.set_color(RED)
		vecxyn2.set_color(YELLOW)
		vecxzn2.set_color(GREEN)
		vecyxn2.set_color(RED)
		vecyyn2.set_color(YELLOW)
		vecyzn2.set_color(GREEN)
		veczxn2.set_color(RED)
		veczyn2.set_color(YELLOW)
		veczzn2.set_color(GREEN)




		rvecxxp1 = Vector([-1,0,0])
		rvecxyp1 = Vector([0,-1,0])
		rvecxzp1 = Vector([0,0,-1])
		rvecyxp1 = Vector([-1,0,0])
		rvecyyp1 = Vector([0,-1,0])
		rvecyzp1 = Vector([0,0,-1])
		rveczxp1 = Vector([-1,0,0])
		rveczyp1 = Vector([0,-1,0])
		rveczzp1 = Vector([0,0,-1])
		rvecxxp1.shift(RIGHT*2)
		rvecxyp1.shift(RIGHT)
		rvecxzp1.shift(RIGHT)
		rvecyxp1.shift(UP)
		rvecyyp1.shift(UP*2)
		rvecyzp1.shift(UP)
		rveczxp1.shift(OUT)
		rveczyp1.shift(OUT)
		rveczzp1.shift(OUT*2)
		rvecxxp1.set_color(RED)
		rvecxyp1.set_color(YELLOW)
		rvecxzp1.set_color(GREEN)
		rvecyxp1.set_color(RED)
		rvecyyp1.set_color(YELLOW)
		rvecyzp1.set_color(GREEN)
		rveczxp1.set_color(RED)
		rveczyp1.set_color(YELLOW)
		rveczzp1.set_color(GREEN)

		rvecxxn1 = Vector([1,0,0])
		rvecxyn1 = Vector([0,1,0])
		rvecxzn1 = Vector([0,0,1])
		rvecyxn1 = Vector([1,0,0])
		rvecyyn1 = Vector([0,1,0])
		rvecyzn1 = Vector([0,0,1])
		rveczxn1 = Vector([1,0,0])
		rveczyn1 = Vector([0,1,0])
		rveczzn1 = Vector([0,0,1])
		rvecxxn1.shift(LEFT*2)
		rvecxyn1.shift(LEFT)
		rvecxzn1.shift(LEFT)
		rvecyxn1.shift(DOWN)
		rvecyyn1.shift(DOWN*2)
		rvecyzn1.shift(DOWN)
		rveczxn1.shift(IN)
		rveczyn1.shift(IN)
		rveczzn1.shift(IN*2)
		rvecxxn1.set_color(RED)
		rvecxyn1.set_color(YELLOW)
		rvecxzn1.set_color(GREEN)
		rvecyxn1.set_color(RED)
		rvecyyn1.set_color(YELLOW)
		rvecyzn1.set_color(GREEN)
		rveczxn1.set_color(RED)
		rveczyn1.set_color(YELLOW)
		rveczzn1.set_color(GREEN)

		rvecxxp2 = Vector([-1,0,0])
		rvecxyp2 = Vector([0,-1,0])
		rvecxzp2 = Vector([0,0,-1])
		rvecyxp2 = Vector([-1,0,0])
		rvecyyp2 = Vector([0,-1,0])
		rvecyzp2 = Vector([0,0,-1])
		rveczxp2 = Vector([-1,0,0])
		rveczyp2 = Vector([0,-1,0])
		rveczzp2 = Vector([0,0,-1])
		rvecxxp2.shift(RIGHT*3)
		rvecxyp2.shift(RIGHT+UP)
		rvecxzp2.shift(RIGHT+OUT)
		rvecyxp2.shift(UP+RIGHT)
		rvecyyp2.shift(UP*3)
		rvecyzp2.shift(UP+OUT)
		rveczxp2.shift(OUT+RIGHT)
		rveczyp2.shift(OUT+UP)
		rveczzp2.shift(OUT*3)
		rvecxxp2.set_color(RED)
		rvecxyp2.set_color(YELLOW)
		rvecxzp2.set_color(GREEN)
		rvecyxp2.set_color(RED)
		rvecyyp2.set_color(YELLOW)
		rvecyzp2.set_color(GREEN)
		rveczxp2.set_color(RED)
		rveczyp2.set_color(YELLOW)
		rveczzp2.set_color(GREEN)

		rvecxxn2 = Vector([1,0,0])
		rvecxyn2 = Vector([0,1,0])
		rvecxzn2 = Vector([0,0,1])
		rvecyxn2 = Vector([1,0,0])
		rvecyyn2 = Vector([0,1,0])
		rvecyzn2 = Vector([0,0,1])
		rveczxn2 = Vector([1,0,0])
		rveczyn2 = Vector([0,1,0])
		rveczzn2 = Vector([0,0,1])
		rvecxxn2.shift(LEFT*3)
		rvecxyn2.shift(LEFT+DOWN)
		rvecxzn2.shift(LEFT+IN)
		rvecyxn2.shift(DOWN+LEFT)
		rvecyyn2.shift(DOWN*3)
		rvecyzn2.shift(DOWN+IN)
		rveczxn2.shift(IN+LEFT)
		rveczyn2.shift(IN+DOWN)
		rveczzn2.shift(IN*3)
		rvecxxn2.set_color(RED)
		rvecxyn2.set_color(YELLOW)
		rvecxzn2.set_color(GREEN)
		rvecyxn2.set_color(RED)
		rvecyyn2.set_color(YELLOW)
		rvecyzn2.set_color(GREEN)
		rveczxn2.set_color(RED)
		rveczyn2.set_color(YELLOW)
		rveczzn2.set_color(GREEN)


		self.set_camera_orientation(phi=75 * DEGREES,theta= 30*DEGREES)
		self.begin_ambient_camera_rotation(rate=0)

		#self.add(axes)
		self.add(prism1)
		# self.add(vecxx,vecxy,vecxz,vecyx,vecyy,vecyz,veczx,veczy,veczz)
		self.wait()

		self.play(
			ApplyMethod(prism1.apply_matrix, matrixxx),
			ReplacementTransform(vecxxp1,vecxxp2),
			ReplacementTransform(vecxxn1,vecxxn2)
		 )
		self.remove(vecxxp2,vecxxn2)
		self.play(
			ApplyMethod(prism1.apply_matrix, rmatrixxx),
			ReplacementTransform(rvecxxp2,rvecxxp1),
			ReplacementTransform(rvecxxn2,rvecxxn1)
		 )
		self.remove(rvecxxp1,rvecxxn1)
		self.play(
			ApplyMethod(prism1.apply_matrix, matrixyy),
			ReplacementTransform(vecyyp1,vecyyp2),
			ReplacementTransform(vecyyn1,vecyyn2)
		 )
		self.remove(vecyyp2,vecyyn2)
		self.play(
			ApplyMethod(prism1.apply_matrix, rmatrixyy),
			ReplacementTransform(rvecyyp2,rvecyyp1),
			ReplacementTransform(rvecyyn2,rvecyyn1)
		 )
		self.remove(rvecyyp1,rvecyyn1)
		self.play(
			ApplyMethod(prism1.apply_matrix, matrixzz),
			ReplacementTransform(veczzp1,veczzp2),
			ReplacementTransform(veczzn1,veczzn2)
		 )
		self.remove(veczzp2,veczzn2)
		self.play(
			ApplyMethod(prism1.apply_matrix, rmatrixzz),
			ReplacementTransform(rveczzp2,rveczzp1),
			ReplacementTransform(rveczzn2,rveczzn1)
		 )
		self.remove(rveczzp1,rveczzn1)
		self.play(
			ApplyMethod(prism1.apply_matrix, matrixzx),
			ReplacementTransform(veczxp1,veczxp2),
			ReplacementTransform(veczxn1,veczxn2)
		 )
		self.remove(veczxp2,veczxn2)
		self.play(
			ApplyMethod(prism1.apply_matrix, rmatrixzx),
			ReplacementTransform(rveczxp2,rveczxp1),
			ReplacementTransform(rveczxn2,rveczxn1)
		 )
		self.remove(rveczxp1,rveczxn1)

		self.play(
			ApplyMethod(prism1.apply_matrix, matrixzy),
			ReplacementTransform(veczyp1,veczyp2),
			ReplacementTransform(veczyn1,veczyn2)
		 )
		self.remove(veczyp2,veczyn2)
		self.play(
			ApplyMethod(prism1.apply_matrix, rmatrixzy),
			ReplacementTransform(rveczyp2,rveczyp1),
			ReplacementTransform(rveczyn2,rveczyn1)
		 )
		self.remove(rveczyp1,rveczyn1)

		self.wait()


class StrainRateTensorDeformationDilatation(Scene):
	def construct(self):
		matrix = Matrix(
			[["\\pdv{u}{x}", "{1 \\over 2}(\\pdv{v}{x}+\\pdv{u}{y})", "{1 \\over 2}(\\pdv{w}{x}+\\pdv{u}{z})"], 
			["{1 \\over 2}(\\pdv{u}{y}+\\pdv{v}{x})", "\\pdv{v}{y}", "{1 \\over 2}(\\pdv{w}{y}+\\pdv{v}{z})"],
			["{1 \\over 2}(\\pdv{u}{z}+\\pdv{w}{x})", "{1 \\over 2}(\\pdv{v}{z}+\\pdv{w}{y})", "\\pdv{w}{z}"]
			], v_buff=1.5, h_buff=3.5)

		text1 = TextMobject("Strain rate tensor, ",
		"dilatation ",
		"part and ",
		"deformation ",
		"part")
		
		text1[0].set_color(YELLOW)
		text1[1].set_color(GREEN)
		text1[3].set_color(PURPLE)

		text = TexMobject("\\underline{\\underline{D}} ",
			"=")

		text[0].set_color(YELLOW)
		matrix[1].set_color(YELLOW)
		matrix[2].set_color(YELLOW)
		matrix[0][0].set_color(GREEN)
		matrix[0][4].set_color(GREEN)
		matrix[0][8].set_color(GREEN)
		matrix[0][1].set_color(PURPLE)
		matrix[0][2].set_color(PURPLE)
		matrix[0][3].set_color(PURPLE)
		matrix[0][5].set_color(PURPLE)
		matrix[0][6].set_color(PURPLE)
		matrix[0][7].set_color(PURPLE)


		text1.to_edge(UP)
		text.shift(LEFT*5)
		matrix.next_to(text)

		self.add(text)
		self.add(matrix)
		self.play(Write(text1[0]))
		self.play(Write(text1[1]),Indicate(matrix[0][0]), Indicate(matrix[0][4]), Indicate(matrix[0][8]) )
		self.play(Write(text1[2]))
		self.play(Write(text1[3]),Indicate(matrix[0][1]), Indicate(matrix[0][2]), Indicate(matrix[0][3]), Indicate(matrix[0][5]), Indicate(matrix[0][6]), Indicate(matrix[0][7])), 
		self.play(Write(text1[4]))
		self.wait(3)


class StrainRateTensorDilatation(Scene):
	def construct(self):
		matrix = Matrix(
			[["\\pdv{u}{x}", "{1 \\over 2}(\\pdv{v}{x}+\\pdv{u}{y})", "{1 \\over 2}(\\pdv{w}{x}+\\pdv{u}{z})"], 
			["{1 \\over 2}(\\pdv{u}{y}+\\pdv{v}{x})", "\\pdv{v}{y}", "{1 \\over 2}(\\pdv{w}{y}+\\pdv{v}{z})"],
			["{1 \\over 2}(\\pdv{u}{z}+\\pdv{w}{x})", "{1 \\over 2}(\\pdv{v}{z}+\\pdv{w}{y})", "\\pdv{w}{z}"]
			], v_buff=1.5, h_buff=3.5)

		text1 = TextMobject("Strain rate tensor",
			", the ",
			"dilatation ",
			"part")
		text = TexMobject("\\underline{\\underline{D}}",
			" =")

		text1[0].set_color(YELLOW)
		text1[2].set_color(GREEN)
		text[0].set_color(YELLOW)
		matrix[1].set_color(YELLOW)
		matrix[2].set_color(YELLOW)
		matrix[0][0].set_color(GREEN)
		matrix[0][4].set_color(GREEN)
		matrix[0][8].set_color(GREEN)


		text1.to_edge(UP)
		text.shift(LEFT*5)
		matrix.next_to(text)

		
		self.add(text)
		self.add(matrix)
		self.play(Write(text1))

		self.wait(4)


class Dilatation(LinearTransformationScene):
	CONFIG = {
		"include_background_plane": False,
		"include_foreground_plane": False,
		"foreground_plane_kwargs": {
			"x_radius": FRAME_WIDTH,
			"y_radius": FRAME_HEIGHT,
			"secondary_line_ratio": 0
		},
		"background_plane_kwargs": {
			"color": GREY,
			"secondary_color": DARK_GREY,
			"axes_color": GREY,
			"stroke_width": 2,
		},
		"show_coordinates": False,
		"show_basis_vectors": False,
		"basis_vector_stroke_width": 6,
		"i_hat_color": X_COLOR,
		"j_hat_color": Y_COLOR,
		"leave_ghost_vectors": False,
	}
	def construct(self):
		rect = Rectangle(height=2, width=4)
		matrix = [[2, 0], [0, 1]]


		text = TextMobject("Dilatation or strech")
		text.to_edge(UP)

		self.play(Write(text))

		self.add_transformable_mobject(rect)
	

		self.apply_matrix(matrix)

		self.wait(3)


class FieldSigmoidScene(MovingCameraScene):
	CONFIG={
		"dot_kwargs":{
			"radius":.05,
			"color":YELLOW,
			"fill_opacity": 1,
		}
	}


	def construct(self):
		FieldSigmoid = VectorField(FuncSigmoidShift, x_max = int(np.ceil(FRAME_WIDTH / 2 + 3)), x_min = int(np.ceil(FRAME_WIDTH / 2 - 12)))
		FieldDiv = VectorField(FuncDivShift)

		title = TextMobject("Dilatation or strech")
		title.add_background_rectangle()
		title.to_edge(UP)
		title.shift(RIGHT*4)


		rect = Rectangle(height=2, width=4)
		matrix = [[4, 0], [0, 1]]
		rect.shift(RIGHT*2.2)


		corner1 = Dot([4,1,0], color=WHITE)
		corner2 = Dot([4,-1,0], color=WHITE)
		corner3 = Dot([0,-1,0], color=WHITE)
		corner4 = Dot([0,1,0], color=WHITE)
		leftdot = Dot([0,0,0], color=RED)
		rightdot = Dot([4,0,0], color=BLUE)

		corners=VGroup()
		for vector in FieldSigmoid:
			dot = Dot(**self.dot_kwargs)
			dot.move_to(vector.get_start())
			dot.target = vector
			corners.add(dot)


		self.play(
		self.camera_frame.move_to, RIGHT*4,
			run_time=0.01
			)
		self.add(title)
		self.add(FieldSigmoid)
		self.wait()
		#self.add(corner1,corner2,corner3,corner4,leftdot,rightdot)
		
		self.add(rect)
		self.wait()
		


		# for dot in corner1,corner2,corner3,corner4,leftdot,rightdot:
		# 	move_submobjects_along_vector_field(
		# 		dot,
		# 		lambda p: FuncSigmoidShift(p)
		# 	)
		self.play(
			ApplyMethod(rect.apply_matrix, matrix), run_time=5
			#ApplyMethod(rect.shift, RIGHT*2)
		)
		# self.wait(4)

		# for dot in corner1,corner2,corner3,corner4,leftdot,rightdot:
		# 	dot.clear_updaters()
		self.wait()

		# self.remove(rect)
		# self.play(ReplacementTransform(FieldSigmoid,FieldDiv))


class FieldDivergenceScene(Scene):
	CONFIG={
		"dot_kwargs":{
			"radius":.05,
			"color":YELLOW,
			"fill_opacity": 1,
		}
	}


	def construct(self):
		FieldDiv = VectorField(FuncDiv)

		title = TextMobject("Dilatation or strech")
		title.add_background_rectangle()
		title.to_edge(UP)

		rect = Rectangle(height=2, width=4)
		matrix = [[6, 0], [0, 6]]

		corner1 = Dot([2,1,0], color=WHITE)
		corner2 = Dot([2,-1,0], color=WHITE)
		corner3 = Dot([-2,-1,0], color=WHITE)
		corner4 = Dot([-2,1,0], color=WHITE)
		leftdot = Dot([-2,0,0], color=RED)
		rightdot = Dot([2,0,0], color=BLUE)

		corners=VGroup()
		for vector in FieldDiv:
			dot = Dot(**self.dot_kwargs)
			dot.move_to(vector.get_start())
			dot.target = vector
			corners.add(dot)


		self.add(FieldDiv)
		self.add(title)
		self.wait()
		# self.add(corner1,corner2,corner3,corner4,leftdot,rightdot)

		# self.wait()
		# for dot in corner1,corner2,corner3,corner4,leftdot,rightdot:
		# 	move_submobjects_along_vector_field(
		# 		dot,
		# 		lambda p: FuncDiv(p)
		# 	)
		self.play(ApplyMethod(rect.apply_matrix, matrix), run_time=5)
		# self.wait(3)

		# for dot in corner1,corner2,corner3,corner4,leftdot,rightdot:
		# 	dot.clear_updaters()


class DilatationThreeD(ThreeDScene):
	def construct(self):
		axes = ThreeDAxes()

		title = TextMobject("Dilatation or strech")
		title.add_background_rectangle()
		title.to_edge(UP)

		prism1 = Prism(dimensions = [4,2,2])
		prism2 = Prism(dimensions = [6,1.8,1.8])
		axes.add(axes.get_axis_labels())

		
		vecxxp1 = Vector([1,0,0])
		vecxxm1 = Vector([-1,0,0])
		vecxxp2 = Vector([1,0,0])
		vecxxm2 = Vector([-1,0,0])
		vecxxp1.shift(RIGHT*2)
		vecxxm1.shift(LEFT*2)
		vecxxp2.shift(RIGHT*3)
		vecxxm2.shift(LEFT*3)
		vecxxp1.set_color(RED)
		vecxxm1.set_color(RED)
		vecxxp2.set_color(RED)
		vecxxm2.set_color(RED)
		# vecxy = Vector([0,1,0])
		# vecxz = Vector([0,0,1])
		# vecyx = Vector([1,0,0])
		# vecyy = Vector([0,1,0])
		# vecyz = Vector([0,0,1])
		# veczx = Vector([1,0,0])
		# veczy = Vector([0,1,0])
		# veczz = Vector([0,0,1])
		# vecxx.shift(RIGHT)
		# vecxy.shift(RIGHT)
		# vecxz.shift(RIGHT)
		# vecyx.shift(UP)
		# vecyy.shift(UP)
		# vecyz.shift(UP)
		# veczx.shift(OUT)
		# veczy.shift(OUT)
		# veczz.shift(OUT)
		# vecxx.set_color(RED)
		# vecxy.set_color(YELLOW)
		# vecxz.set_color(GREEN)
		# vecyx.set_color(RED)
		# vecyy.set_color(YELLOW)
		# vecyz.set_color(GREEN)
		# veczx.set_color(RED)
		# veczy.set_color(YELLOW)
		# veczz.set_color(GREEN)

		# text = TextMobject("Dilatation or streching")
		# text.to_edge(UP)

		# self.play(Write(text))
		# self.wait()
		# self.remove(text)


		self.set_camera_orientation(phi=75 * DEGREES)
		self.begin_ambient_camera_rotation(rate=0.2)

		# self.add(axes)
		self.add_fixed_in_frame_mobjects(title)
		self.add(prism1)
		# self.add(vecxx,vecxy,vecxz,vecyx,vecyy,vecyz,veczx,veczy,veczz)
		self.wait()

		self.play(
			#ReplacementTransform(vecxxp1,vecxxp2),
			#ReplacementTransform(vecxxm1,vecxxm2),
			ReplacementTransform(prism1,prism2)
		)
		#self.remove(vecxxp2,vecxxm2)
		self.wait(2)


class StrainRateTensorDilatationDivergence(Scene):
	def construct(self):
		matrix = Matrix(
			[["\\pdv{u}{x}", "{1 \\over 2}(\\pdv{v}{x}+\\pdv{u}{y})", "{1 \\over 2}(\\pdv{w}{x}+\\pdv{u}{z})"], 
			["{1 \\over 2}(\\pdv{u}{y}+\\pdv{v}{x})", "\\pdv{v}{y}", "{1 \\over 2}(\\pdv{w}{y}+\\pdv{v}{z})"],
			["{1 \\over 2}(\\pdv{u}{z}+\\pdv{w}{x})", "{1 \\over 2}(\\pdv{v}{z}+\\pdv{w}{y})", "\\pdv{w}{z}"]
			], v_buff=1.5, h_buff=3.5)

		text1 = TextMobject("The ",
		"dilatation ",
		"part of the ",
		"strain rate tensor")
		text2 = TextMobject("shows ", "divergence")

		text1[1].set_color(GREEN)
		text = TexMobject("\\underline{\\underline{D}}",
			" =")

		text2[1].set_color(ORANGE)
		text1[1].set_color(GREEN)
		text1[3].set_color(YELLOW)
		text[0].set_color(YELLOW)
		matrix[1].set_color(YELLOW)
		matrix[2].set_color(YELLOW)
		matrix[0][0].set_color(GREEN)
		matrix[0][4].set_color(GREEN)
		matrix[0][8].set_color(GREEN)


		text1.to_edge(UP)
		text2.next_to(text1, DOWN)
		text.shift(LEFT*5)
		matrix.next_to(text)

		Div = Matrix(["\\pdv{u}{x}","\\pdv{v}{y}","\\pdv{w}{z}"], v_buff=1.5)
		Div[0][0].set_color(GREEN)
		Div[0][1].set_color(GREEN)
		Div[0][2].set_color(GREEN)
		Div[1].set_color(ORANGE)
		Div[2].set_color(ORANGE)

		textDiv = TexMobject("= ", "\\div{","\\textbf{v}}")
		textDiv[1].set_color(ORANGE)
		Div.shift(DOWN)
		textDiv.next_to(Div, RIGHT)

		text3b1b = TextMobject("3Blue",
			"1Brown")
		text3b1b[0].set_color(BLUE)
		text3b1b[1].set_color("#964b00")



		self.play(Write(text1))
		self.play(Write(text2))
		self.play(Write(text))
		self.play(Write(matrix))
		self.wait(2)

		self.remove(text, matrix)
		self.play(
			ReplacementTransform(matrix[0][0],Div[0]),
			ReplacementTransform(matrix[0][4],Div[1]),
			ReplacementTransform(matrix[0][8],Div[2])
		)
		self.wait()

		self.play(Write(textDiv))
		self.wait()

		self.play(
			FadeOut(Div), FadeOut(text1), FadeOut(text2),
			ReplacementTransform(textDiv,text3b1b)
		)
		self.wait()


class FieldDivergenceExplanationScene(Scene):
	CONFIG={
		"dot_kwargs":{
			"radius":.05,
			"color":YELLOW,
			"fill_opacity": 1,
		}
	}

	def construct(self):
		FieldDiv = VectorField(FuncDiv)

		circ = Circle(radius=2, color = WHITE)
		matrix = [[4, 0], [0, 1]]


		dots=VGroup()
		for vector in FieldDiv:
			dot = Dot(**self.dot_kwargs)
			dot.set_color(WHITE)
			dot.move_to(vector.get_start())
			dot.target = vector
			dots.add(dot)

		title = TextMobject("Divergence")
		title.scale(1)
		title.add_background_rectangle()
		title.to_edge(UP)

		
		
		self.add(FieldDiv)
		self.play(Write(title))
		self.wait()
				
		self.play(ShowCreation(circ))
		self.wait()

		self.play(
			ShowCreation(dots),
			#run_time=2,
			)
		self.wait()

		move_submobjects_along_vector_field(dots, FuncDiv)
		self.wait(3)
		self.remove(dots)
		self.wait()
		

class StrainRateTensorDeformation(Scene):
	def construct(self):
		matrix = Matrix(
			[["\\pdv{u}{x}", "{1 \\over 2}(\\pdv{v}{x}+\\pdv{u}{y})", "{1 \\over 2}(\\pdv{w}{x}+\\pdv{u}{z})"], 
			["{1 \\over 2}(\\pdv{u}{y}+\\pdv{v}{x})", "\\pdv{v}{y}", "{1 \\over 2}(\\pdv{w}{y}+\\pdv{v}{z})"],
			["{1 \\over 2}(\\pdv{u}{z}+\\pdv{w}{x})", "{1 \\over 2}(\\pdv{v}{z}+\\pdv{w}{y})", "\\pdv{w}{z}"]
			], v_buff=1.5, h_buff=3.5)

		text1 = TextMobject("Strain rate tensor",
			", the ",
			"deformation ",
			"part")
		text = TexMobject("\\underline{\\underline{D}} ",
			"=")

		text1[0].set_color(YELLOW)
		text1[2].set_color(PURPLE)
		text[0].set_color(YELLOW)
		matrix[1].set_color(YELLOW)
		matrix[2].set_color(YELLOW)
		matrix[0][1].set_color(PURPLE)
		matrix[0][2].set_color(PURPLE)
		matrix[0][3].set_color(PURPLE)
		matrix[0][5].set_color(PURPLE)
		matrix[0][6].set_color(PURPLE)
		matrix[0][7].set_color(PURPLE)


		text1.to_edge(UP)
		text.shift(LEFT*5)
		matrix.next_to(text)

		self.add(text)
		self.add(matrix)
		self.play(Write(text1))

		self.wait(3)


class Deformation(LinearTransformationScene):
	CONFIG = {
		"include_background_plane": False,
		"include_foreground_plane": False,
		"foreground_plane_kwargs": {
			"x_radius": FRAME_WIDTH,
			"y_radius": FRAME_HEIGHT,
			"secondary_line_ratio": 0
		},
		"background_plane_kwargs": {
			"color": GREY,
			"secondary_color": DARK_GREY,
			"axes_color": GREY,
			"stroke_width": 2,
		},
		"show_coordinates": False,
		"show_basis_vectors": False,
		"basis_vector_stroke_width": 6,
		"i_hat_color": X_COLOR,
		"j_hat_color": Y_COLOR,
		"leave_ghost_vectors": False,
	}
	def construct(self):
		rect = Rectangle(height=2, width=4)
		rect.move_to(0)
		vector_array = np.array([[1], [2]])
		matrix = [[1, 1], [0, 1]]

		text = TextMobject("Deformation or strain")
		text.to_edge(UP)

		self.play(Write(text))

		self.add_transformable_mobject(rect)

		self.apply_matrix(matrix)

		self.wait(3)


class FieldDeformationScene(Scene):
	CONFIG={
		"dot_kwargs":{
			"radius":.05,
			"color":YELLOW,
			"fill_opacity": 1,
		}
	}


	def construct(self):
		FieldDeform = VectorField(FuncDeform)

		title = TextMobject("Deformation or strain")
		title.add_background_rectangle()
		title.to_edge(UP)

		rect = Rectangle(height=2, width=4)
		matrix = [[1, 2], [0, 1]]

		corner1 = Dot([2,1,0], color=WHITE)
		corner2 = Dot([2,-1,0], color=WHITE)
		corner3 = Dot([-2,-1,0], color=WHITE)
		corner4 = Dot([-2,1,0], color=WHITE)
		leftdot = Dot([-2,-1,0], color=RED)
		rightdot = Dot([2,1,0], color=BLUE)

		corners=VGroup()
		for vector in FieldDeform:
			dot = Dot(**self.dot_kwargs)
			dot.move_to(vector.get_start())
			dot.target = vector
			corners.add(dot)


		self.add(FieldDeform)
		self.add(title)
		self.wait()
		# self.add(corner1,corner2,corner3,corner4,leftdot,rightdot)

		# self.wait()
		# for dot in corner1,corner2,corner3,corner4,leftdot,rightdot:
		# 	move_submobjects_along_vector_field(
		# 		dot,
		# 		lambda p: FuncDeform(p)
		# 	)
		self.play(ApplyMethod(rect.apply_matrix, matrix), run_time=5)
		# self.wait(3)

		# for dot in corner1,corner2,corner3,corner4,leftdot,rightdot:
		# 	dot.clear_updaters()


class DeformationThreeD(ThreeDScene):
	def construct(self):
		axes = ThreeDAxes()
		prism1 = Prism(dimensions = [4,2,2])
		axes.add(axes.get_axis_labels())

		title = TextMobject("Deformation or strain")
		title.add_background_rectangle()
		title.to_edge(UP)

		matrixxx = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
		matrixxy = [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
		matrixxz = [[1, 0, 0], [0, 1, 0], [1, 0, 1]]

		matrixyy = [[1, 0, 0], [0, 2, 0], [0, 0, 1]]
		matrixyx = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
		matrixyz = [[1, 0, 0], [0, 1, 0], [0, 1, 1]]

		matrixzz = [[1, 0, 0], [0, 1, 0], [0, 0, 2]]
		matrixzx = [[1, 0, 1], [0, 1, 0], [0, 0, 1]]
		matrixzy = [[1, 0, 0], [0, 1, 1], [0, 0, 1]]


		rmatrixxx = [[1/2, 0, 0], [0, 1, 0], [0, 0, 1]]
		rmatrixxy = [[1, 0, 0], [-1, 1, 0], [0, 0, 1]]
		rmatrixxz = [[1, 0, 0], [0, 1, 0], [-1, 0, 1]]

		rmatrixyy = [[1, 0, 0], [0, 1/2, 0], [0, 0, 1]]
		rmatrixyx = [[1, -1, 0], [0, 1, 0], [0, 0, 1]]
		rmatrixyz = [[1, 0, 0], [0, 1, 0], [0, -1, 1]]

		rmatrixzz = [[1, 0, 0], [0, 1, 0], [0, 0, 1/2]]
		rmatrixzx = [[1, 0, -1], [0, 1, 0], [0, 0, 1]]
		rmatrixzy = [[1, 0, 0], [0, 1, -1], [0, 0, 1]]

		
		vecxxp1 = Vector([1,0,0])
		vecxyp1 = Vector([0,1,0])
		vecxzp1 = Vector([0,0,1])
		vecyxp1 = Vector([1,0,0])
		vecyyp1 = Vector([0,1,0])
		vecyzp1 = Vector([0,0,1])
		veczxp1 = Vector([1,0,0])
		veczyp1 = Vector([0,1,0])
		veczzp1 = Vector([0,0,1])
		vecxxp1.shift(RIGHT)
		vecxyp1.shift(RIGHT)
		vecxzp1.shift(RIGHT)
		vecyxp1.shift(UP)
		vecyyp1.shift(UP)
		vecyzp1.shift(UP)
		veczxp1.shift(OUT)
		veczyp1.shift(OUT)
		veczzp1.shift(OUT)
		vecxxp1.set_color(RED)
		vecxyp1.set_color(YELLOW)
		vecxzp1.set_color(GREEN)
		vecyxp1.set_color(RED)
		vecyyp1.set_color(YELLOW)
		vecyzp1.set_color(GREEN)
		veczxp1.set_color(RED)
		veczyp1.set_color(YELLOW)
		veczzp1.set_color(GREEN)

		vecxxn1 = Vector([-1,0,0])
		vecxyn1 = Vector([0,-1,0])
		vecxzn1 = Vector([0,0,-1])
		vecyxn1 = Vector([-1,0,0])
		vecyyn1 = Vector([0,-1,0])
		vecyzn1 = Vector([0,0,-1])
		veczxn1 = Vector([-1,0,0])
		veczyn1 = Vector([0,-1,0])
		veczzn1 = Vector([0,0,-1])
		vecxxn1.shift(LEFT)
		vecxyn1.shift(LEFT)
		vecxzn1.shift(LEFT)
		vecyxn1.shift(DOWN)
		vecyyn1.shift(DOWN)
		vecyzn1.shift(DOWN)
		veczxn1.shift(IN)
		veczyn1.shift(IN)
		veczzn1.shift(IN)
		vecxxn1.set_color(RED)
		vecxyn1.set_color(YELLOW)
		vecxzn1.set_color(GREEN)
		vecyxn1.set_color(RED)
		vecyyn1.set_color(YELLOW)
		vecyzn1.set_color(GREEN)
		veczxn1.set_color(RED)
		veczyn1.set_color(YELLOW)
		veczzn1.set_color(GREEN)

		vecxxp2 = Vector([1,0,0])
		vecxyp2 = Vector([0,1,0])
		vecxzp2 = Vector([0,0,1])
		vecyxp2 = Vector([1,0,0])
		vecyyp2 = Vector([0,1,0])
		vecyzp2 = Vector([0,0,1])
		veczxp2 = Vector([1,0,0])
		veczyp2 = Vector([0,1,0])
		veczzp2 = Vector([0,0,1])
		vecxxp2.shift(RIGHT*2)
		vecxyp2.shift(RIGHT+UP)
		vecxzp2.shift(RIGHT+OUT)
		vecyxp2.shift(UP+RIGHT)
		vecyyp2.shift(UP*2)
		vecyzp2.shift(UP+OUT)
		veczxp2.shift(OUT+RIGHT)
		veczyp2.shift(OUT+UP)
		veczzp2.shift(OUT*2)
		vecxxp2.set_color(RED)
		vecxyp2.set_color(YELLOW)
		vecxzp2.set_color(GREEN)
		vecyxp2.set_color(RED)
		vecyyp2.set_color(YELLOW)
		vecyzp2.set_color(GREEN)
		veczxp2.set_color(RED)
		veczyp2.set_color(YELLOW)
		veczzp2.set_color(GREEN)

		vecxxn2 = Vector([-1,0,0])
		vecxyn2 = Vector([0,-1,0])
		vecxzn2 = Vector([0,0,-1])
		vecyxn2 = Vector([-1,0,0])
		vecyyn2 = Vector([0,-1,0])
		vecyzn2 = Vector([0,0,-1])
		veczxn2 = Vector([-1,0,0])
		veczyn2 = Vector([0,-1,0])
		veczzn2 = Vector([0,0,-1])
		vecxxn2.shift(LEFT*2)
		vecxyn2.shift(LEFT+DOWN)
		vecxzn2.shift(LEFT+IN)
		vecyxn2.shift(DOWN+LEFT)
		vecyyn2.shift(DOWN*2)
		vecyzn2.shift(DOWN+IN)
		veczxn2.shift(IN+LEFT)
		veczyn2.shift(IN+DOWN)
		veczzn2.shift(IN*2)
		vecxxn2.set_color(RED)
		vecxyn2.set_color(YELLOW)
		vecxzn2.set_color(GREEN)
		vecyxn2.set_color(RED)
		vecyyn2.set_color(YELLOW)
		vecyzn2.set_color(GREEN)
		veczxn2.set_color(RED)
		veczyn2.set_color(YELLOW)
		veczzn2.set_color(GREEN)




		rvecxxp1 = Vector([-1,0,0])
		rvecxyp1 = Vector([0,-1,0])
		rvecxzp1 = Vector([0,0,-1])
		rvecyxp1 = Vector([-1,0,0])
		rvecyyp1 = Vector([0,-1,0])
		rvecyzp1 = Vector([0,0,-1])
		rveczxp1 = Vector([-1,0,0])
		rveczyp1 = Vector([0,-1,0])
		rveczzp1 = Vector([0,0,-1])
		rvecxxp1.shift(RIGHT*2)
		rvecxyp1.shift(RIGHT)
		rvecxzp1.shift(RIGHT)
		rvecyxp1.shift(UP)
		rvecyyp1.shift(UP*2)
		rvecyzp1.shift(UP)
		rveczxp1.shift(OUT)
		rveczyp1.shift(OUT)
		rveczzp1.shift(OUT*2)
		rvecxxp1.set_color(RED)
		rvecxyp1.set_color(YELLOW)
		rvecxzp1.set_color(GREEN)
		rvecyxp1.set_color(RED)
		rvecyyp1.set_color(YELLOW)
		rvecyzp1.set_color(GREEN)
		rveczxp1.set_color(RED)
		rveczyp1.set_color(YELLOW)
		rveczzp1.set_color(GREEN)

		rvecxxn1 = Vector([1,0,0])
		rvecxyn1 = Vector([0,1,0])
		rvecxzn1 = Vector([0,0,1])
		rvecyxn1 = Vector([1,0,0])
		rvecyyn1 = Vector([0,1,0])
		rvecyzn1 = Vector([0,0,1])
		rveczxn1 = Vector([1,0,0])
		rveczyn1 = Vector([0,1,0])
		rveczzn1 = Vector([0,0,1])
		rvecxxn1.shift(LEFT*2)
		rvecxyn1.shift(LEFT)
		rvecxzn1.shift(LEFT)
		rvecyxn1.shift(DOWN)
		rvecyyn1.shift(DOWN*2)
		rvecyzn1.shift(DOWN)
		rveczxn1.shift(IN)
		rveczyn1.shift(IN)
		rveczzn1.shift(IN*2)
		rvecxxn1.set_color(RED)
		rvecxyn1.set_color(YELLOW)
		rvecxzn1.set_color(GREEN)
		rvecyxn1.set_color(RED)
		rvecyyn1.set_color(YELLOW)
		rvecyzn1.set_color(GREEN)
		rveczxn1.set_color(RED)
		rveczyn1.set_color(YELLOW)
		rveczzn1.set_color(GREEN)

		rvecxxp2 = Vector([-1,0,0])
		rvecxyp2 = Vector([0,-1,0])
		rvecxzp2 = Vector([0,0,-1])
		rvecyxp2 = Vector([-1,0,0])
		rvecyyp2 = Vector([0,-1,0])
		rvecyzp2 = Vector([0,0,-1])
		rveczxp2 = Vector([-1,0,0])
		rveczyp2 = Vector([0,-1,0])
		rveczzp2 = Vector([0,0,-1])
		rvecxxp2.shift(RIGHT*3)
		rvecxyp2.shift(RIGHT+UP)
		rvecxzp2.shift(RIGHT+OUT)
		rvecyxp2.shift(UP+RIGHT)
		rvecyyp2.shift(UP*3)
		rvecyzp2.shift(UP+OUT)
		rveczxp2.shift(OUT+RIGHT)
		rveczyp2.shift(OUT+UP)
		rveczzp2.shift(OUT*3)
		rvecxxp2.set_color(RED)
		rvecxyp2.set_color(YELLOW)
		rvecxzp2.set_color(GREEN)
		rvecyxp2.set_color(RED)
		rvecyyp2.set_color(YELLOW)
		rvecyzp2.set_color(GREEN)
		rveczxp2.set_color(RED)
		rveczyp2.set_color(YELLOW)
		rveczzp2.set_color(GREEN)

		rvecxxn2 = Vector([1,0,0])
		rvecxyn2 = Vector([0,1,0])
		rvecxzn2 = Vector([0,0,1])
		rvecyxn2 = Vector([1,0,0])
		rvecyyn2 = Vector([0,1,0])
		rvecyzn2 = Vector([0,0,1])
		rveczxn2 = Vector([1,0,0])
		rveczyn2 = Vector([0,1,0])
		rveczzn2 = Vector([0,0,1])
		rvecxxn2.shift(LEFT*3)
		rvecxyn2.shift(LEFT+DOWN)
		rvecxzn2.shift(LEFT+IN)
		rvecyxn2.shift(DOWN+LEFT)
		rvecyyn2.shift(DOWN*3)
		rvecyzn2.shift(DOWN+IN)
		rveczxn2.shift(IN+LEFT)
		rveczyn2.shift(IN+DOWN)
		rveczzn2.shift(IN*3)
		rvecxxn2.set_color(RED)
		rvecxyn2.set_color(YELLOW)
		rvecxzn2.set_color(GREEN)
		rvecyxn2.set_color(RED)
		rvecyyn2.set_color(YELLOW)
		rvecyzn2.set_color(GREEN)
		rveczxn2.set_color(RED)
		rveczyn2.set_color(YELLOW)
		rveczzn2.set_color(GREEN)


		# text = TextMobject("Deformation")
		# text.to_edge(UP)

		# self.play(Write(text))
		# self.wait()
		# self.remove(text)


		self.set_camera_orientation(phi=75 * DEGREES,theta= 210*DEGREES)
		self.begin_ambient_camera_rotation(rate=0.2)

		#self.add(axes)
		self.add_fixed_in_frame_mobjects(title)
		self.add(prism1)
		# self.add(vecxx,vecxy,vecxz,vecyx,vecyy,vecyz,veczx,veczy,veczz)
		self.wait()

		self.play(
			ApplyMethod(prism1.apply_matrix, matrixzx),
			#ReplacementTransform(veczxp1,veczxp2),
			#ReplacementTransform(veczxn1,veczxn2), 
			run_time=2
		 )
		#self.remove(veczxp2,veczxn2)
		self.wait(2)


class RotationTensor(Scene):
	def construct(self):
		matrix = Matrix(
			[["0", "{1 \\over 2}(\\pdv{v}{x}-\\pdv{u}{y})", "{1 \\over 2}(\\pdv{w}{x}-\\pdv{u}{z})"], 
			["{1 \\over 2}(\\pdv{u}{y}-\\pdv{v}{x})", "0", "{1 \\over 2}(\\pdv{w}{y}-\\pdv{v}{z})"],
			["{1 \\over 2}(\\pdv{u}{z}-\\pdv{w}{x})", "{1 \\over 2}(\\pdv{v}{z}-\\pdv{w}{y})", "0"]
			], v_buff=1.5, h_buff=3.5)

		text1 = TextMobject("Rotation or spin tensor")
		text = TexMobject("\\underline{\\underline{\\Omega}} ",
			"=")

		text1.set_color(BLUE)
		text[0].set_color(BLUE)
		matrix[1].set_color(BLUE)
		matrix[2].set_color(BLUE)
		text1.to_edge(UP)
		text.shift(LEFT*5)
		matrix.next_to(text)

		self.play(Write(text1))
		self.play(Write(text))
		self.play(Write(matrix))

		self.wait(6)


class Rotation(LinearTransformationScene):
	CONFIG = {
		"include_background_plane": False,
		"include_foreground_plane": False,
		"foreground_plane_kwargs": {
			"x_radius": FRAME_WIDTH,
			"y_radius": FRAME_HEIGHT,
			"secondary_line_ratio": 0
		},
		"background_plane_kwargs": {
			"color": GREY,
			"secondary_color": DARK_GREY,
			"axes_color": GREY,
			"stroke_width": 2,
		},
		"show_coordinates": False,
		"show_basis_vectors": False,
		"basis_vector_stroke_width": 6,
		"i_hat_color": X_COLOR,
		"j_hat_color": Y_COLOR,
		"leave_ghost_vectors": False,
	}
	def construct(self):
		mob = Rectangle(height=2, width=4)
		vector_array = np.array([[-2], [1]])
		matrix = [[np.cos(1 * TAU/16), -np.cos(PI/4+1 * TAU/16)], [np.sin(1 * TAU / 16), np.sin(PI/4+1 * TAU / 16)]]
		text = TextMobject("Rotation or spin")
		text.to_edge(UP)

		self.play(Write(text))



		self.add_transformable_mobject(mob)
		self.apply_matrix(matrix)

		self.wait(1)


class FieldCurlScene(Scene):
	CONFIG={
		"dot_kwargs":{
			"radius":.05,
			"color":YELLOW,
			"fill_opacity": 1,
		}
	}


	def construct(self):
		FieldCurl = VectorField(FuncCurl)

		rect = Rectangle(height=2, width=4)
		matrix = [[np.cos(1 * TAU/16), -np.cos(PI/4+1 * TAU/16)], [np.sin(1 * TAU / 16), np.sin(PI/4+1 * TAU / 16)]]

		corner1 = Dot([2,1,0], color=WHITE)
		corner2 = Dot([2,-1,0], color=WHITE)
		corner3 = Dot([-2,-1,0], color=WHITE)
		corner4 = Dot([-2,1,0], color=WHITE)
		leftdot = Dot([-2,0,0], color=RED)
		rightdot = Dot([2,0,0], color=BLUE)

		corners=VGroup()
		for vector in FieldCurl:
			dot = Dot(**self.dot_kwargs)
			dot.move_to(vector.get_start())
			dot.target = vector
			corners.add(dot)

		title = TextMobject("Rotation or spin")
		title.add_background_rectangle()
		title.to_edge(UP)


		self.add(FieldCurl)
		self.add(title)
		self.wait()
		# self.add(FieldCurl,corner1,corner2,corner3,corner4,leftdot,rightdot)

		# self.wait()
		# for dot in corner1,corner2,corner3,corner4,leftdot,rightdot:
		# 	move_submobjects_along_vector_field(
		# 		dot,
		# 		lambda p: FuncCurl(p)
		# 	)
		self.play(Rotate(rect, 90*DEGREES), run_time=5)
		# self.wait(3)

		# for dot in corner1,corner2,corner3,corner4,leftdot,rightdot:
		# 	dot.clear_updaters()


class RotationThreeD(ThreeDScene):
	def construct(self):
		axes = ThreeDAxes()
		prism1 = Prism(dimensions = [4,2,2])
		axes.add(axes.get_axis_labels())

		title = TextMobject("Rotation or spin")
		title.add_background_rectangle()
		title.to_edge(UP)

		matrixrot = [[np.cos(1 * TAU/16), -np.cos(PI/4+1 * TAU/16),0], [np.sin(1 * TAU / 16), np.sin(PI/4+1 * TAU / 16),0],[0,0,1]]

		matrixxx = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
		matrixxy = [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
		matrixxz = [[1, 0, 0], [0, 1, 0], [1, 0, 1]]

		matrixyy = [[1, 0, 0], [0, 2, 0], [0, 0, 1]]
		matrixyx = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
		matrixyz = [[1, 0, 0], [0, 1, 0], [0, 1, 1]]

		matrixzz = [[1, 0, 0], [0, 1, 0], [0, 0, 2]]
		matrixzx = [[1, 0, 1], [0, 1, 0], [0, 0, 1]]
		matrixzy = [[1, 0, 0], [0, 1, 1], [0, 0, 1]]


		rmatrixxx = [[1/2, 0, 0], [0, 1, 0], [0, 0, 1]]
		rmatrixxy = [[1, 0, 0], [-1, 1, 0], [0, 0, 1]]
		rmatrixxz = [[1, 0, 0], [0, 1, 0], [-1, 0, 1]]

		rmatrixyy = [[1, 0, 0], [0, 1/2, 0], [0, 0, 1]]
		rmatrixyx = [[1, -1, 0], [0, 1, 0], [0, 0, 1]]
		rmatrixyz = [[1, 0, 0], [0, 1, 0], [0, -1, 1]]

		rmatrixzz = [[1, 0, 0], [0, 1, 0], [0, 0, 1/2]]
		rmatrixzx = [[1, 0, -1], [0, 1, 0], [0, 0, 1]]
		rmatrixzy = [[1, 0, 0], [0, 1, -1], [0, 0, 1]]

		
		vecxxp1 = Vector([1,0,0])
		vecxyp1 = Vector([0,1,0])
		vecxzp1 = Vector([0,0,1])
		vecyxp1 = Vector([1,0,0])
		vecyyp1 = Vector([0,1,0])
		vecyzp1 = Vector([0,0,1])
		veczxp1 = Vector([1,0,0])
		veczyp1 = Vector([0,1,0])
		veczzp1 = Vector([0,0,1])
		vecxxp1.shift(RIGHT)
		vecxyp1.shift(RIGHT)
		vecxzp1.shift(RIGHT)
		vecyxp1.shift(UP)
		vecyyp1.shift(UP)
		vecyzp1.shift(UP)
		veczxp1.shift(OUT)
		veczyp1.shift(OUT)
		veczzp1.shift(OUT)
		vecxxp1.set_color(RED)
		vecxyp1.set_color(YELLOW)
		vecxzp1.set_color(GREEN)
		vecyxp1.set_color(RED)
		vecyyp1.set_color(YELLOW)
		vecyzp1.set_color(GREEN)
		veczxp1.set_color(RED)
		veczyp1.set_color(YELLOW)
		veczzp1.set_color(GREEN)

		vecxxn1 = Vector([-1,0,0])
		vecxyn1 = Vector([0,-1,0])
		vecxzn1 = Vector([0,0,-1])
		vecyxn1 = Vector([-1,0,0])
		vecyyn1 = Vector([0,-1,0])
		vecyzn1 = Vector([0,0,-1])
		veczxn1 = Vector([-1,0,0])
		veczyn1 = Vector([0,-1,0])
		veczzn1 = Vector([0,0,-1])
		vecxxn1.shift(LEFT)
		vecxyn1.shift(LEFT)
		vecxzn1.shift(LEFT)
		vecyxn1.shift(DOWN)
		vecyyn1.shift(DOWN)
		vecyzn1.shift(DOWN)
		veczxn1.shift(IN)
		veczyn1.shift(IN)
		veczzn1.shift(IN)
		vecxxn1.set_color(RED)
		vecxyn1.set_color(YELLOW)
		vecxzn1.set_color(GREEN)
		vecyxn1.set_color(RED)
		vecyyn1.set_color(YELLOW)
		vecyzn1.set_color(GREEN)
		veczxn1.set_color(RED)
		veczyn1.set_color(YELLOW)
		veczzn1.set_color(GREEN)

		vecxxp2 = Vector([1,0,0])
		vecxyp2 = Vector([0,1,0])
		vecxzp2 = Vector([0,0,1])
		vecyxp2 = Vector([1,0,0])
		vecyyp2 = Vector([0,1,0])
		vecyzp2 = Vector([0,0,1])
		veczxp2 = Vector([1,0,0])
		veczyp2 = Vector([0,1,0])
		veczzp2 = Vector([0,0,1])
		vecxxp2.shift(RIGHT*2)
		vecxyp2.shift(RIGHT+UP)
		vecxzp2.shift(RIGHT+OUT)
		vecyxp2.shift(UP+RIGHT)
		vecyyp2.shift(UP*2)
		vecyzp2.shift(UP+OUT)
		veczxp2.shift(OUT+RIGHT)
		veczyp2.shift(OUT+UP)
		veczzp2.shift(OUT*2)
		vecxxp2.set_color(RED)
		vecxyp2.set_color(YELLOW)
		vecxzp2.set_color(GREEN)
		vecyxp2.set_color(RED)
		vecyyp2.set_color(YELLOW)
		vecyzp2.set_color(GREEN)
		veczxp2.set_color(RED)
		veczyp2.set_color(YELLOW)
		veczzp2.set_color(GREEN)

		vecxxn2 = Vector([-1,0,0])
		vecxyn2 = Vector([0,-1,0])
		vecxzn2 = Vector([0,0,-1])
		vecyxn2 = Vector([-1,0,0])
		vecyyn2 = Vector([0,-1,0])
		vecyzn2 = Vector([0,0,-1])
		veczxn2 = Vector([-1,0,0])
		veczyn2 = Vector([0,-1,0])
		veczzn2 = Vector([0,0,-1])
		vecxxn2.shift(LEFT*2)
		vecxyn2.shift(LEFT+DOWN)
		vecxzn2.shift(LEFT+IN)
		vecyxn2.shift(DOWN+LEFT)
		vecyyn2.shift(DOWN*2)
		vecyzn2.shift(DOWN+IN)
		veczxn2.shift(IN+LEFT)
		veczyn2.shift(IN+DOWN)
		veczzn2.shift(IN*2)
		vecxxn2.set_color(RED)
		vecxyn2.set_color(YELLOW)
		vecxzn2.set_color(GREEN)
		vecyxn2.set_color(RED)
		vecyyn2.set_color(YELLOW)
		vecyzn2.set_color(GREEN)
		veczxn2.set_color(RED)
		veczyn2.set_color(YELLOW)
		veczzn2.set_color(GREEN)




		rvecxxp1 = Vector([-1,0,0])
		rvecxyp1 = Vector([0,-1,0])
		rvecxzp1 = Vector([0,0,-1])
		rvecyxp1 = Vector([-1,0,0])
		rvecyyp1 = Vector([0,-1,0])
		rvecyzp1 = Vector([0,0,-1])
		rveczxp1 = Vector([-1,0,0])
		rveczyp1 = Vector([0,-1,0])
		rveczzp1 = Vector([0,0,-1])
		rvecxxp1.shift(RIGHT*2)
		rvecxyp1.shift(RIGHT)
		rvecxzp1.shift(RIGHT)
		rvecyxp1.shift(UP)
		rvecyyp1.shift(UP*2)
		rvecyzp1.shift(UP)
		rveczxp1.shift(OUT)
		rveczyp1.shift(OUT)
		rveczzp1.shift(OUT*2)
		rvecxxp1.set_color(RED)
		rvecxyp1.set_color(YELLOW)
		rvecxzp1.set_color(GREEN)
		rvecyxp1.set_color(RED)
		rvecyyp1.set_color(YELLOW)
		rvecyzp1.set_color(GREEN)
		rveczxp1.set_color(RED)
		rveczyp1.set_color(YELLOW)
		rveczzp1.set_color(GREEN)

		rvecxxn1 = Vector([1,0,0])
		rvecxyn1 = Vector([0,1,0])
		rvecxzn1 = Vector([0,0,1])
		rvecyxn1 = Vector([1,0,0])
		rvecyyn1 = Vector([0,1,0])
		rvecyzn1 = Vector([0,0,1])
		rveczxn1 = Vector([1,0,0])
		rveczyn1 = Vector([0,1,0])
		rveczzn1 = Vector([0,0,1])
		rvecxxn1.shift(LEFT*2)
		rvecxyn1.shift(LEFT)
		rvecxzn1.shift(LEFT)
		rvecyxn1.shift(DOWN)
		rvecyyn1.shift(DOWN*2)
		rvecyzn1.shift(DOWN)
		rveczxn1.shift(IN)
		rveczyn1.shift(IN)
		rveczzn1.shift(IN*2)
		rvecxxn1.set_color(RED)
		rvecxyn1.set_color(YELLOW)
		rvecxzn1.set_color(GREEN)
		rvecyxn1.set_color(RED)
		rvecyyn1.set_color(YELLOW)
		rvecyzn1.set_color(GREEN)
		rveczxn1.set_color(RED)
		rveczyn1.set_color(YELLOW)
		rveczzn1.set_color(GREEN)

		rvecxxp2 = Vector([-1,0,0])
		rvecxyp2 = Vector([0,-1,0])
		rvecxzp2 = Vector([0,0,-1])
		rvecyxp2 = Vector([-1,0,0])
		rvecyyp2 = Vector([0,-1,0])
		rvecyzp2 = Vector([0,0,-1])
		rveczxp2 = Vector([-1,0,0])
		rveczyp2 = Vector([0,-1,0])
		rveczzp2 = Vector([0,0,-1])
		rvecxxp2.shift(RIGHT*3)
		rvecxyp2.shift(RIGHT+UP)
		rvecxzp2.shift(RIGHT+OUT)
		rvecyxp2.shift(UP+RIGHT)
		rvecyyp2.shift(UP*3)
		rvecyzp2.shift(UP+OUT)
		rveczxp2.shift(OUT+RIGHT)
		rveczyp2.shift(OUT+UP)
		rveczzp2.shift(OUT*3)
		rvecxxp2.set_color(RED)
		rvecxyp2.set_color(YELLOW)
		rvecxzp2.set_color(GREEN)
		rvecyxp2.set_color(RED)
		rvecyyp2.set_color(YELLOW)
		rvecyzp2.set_color(GREEN)
		rveczxp2.set_color(RED)
		rveczyp2.set_color(YELLOW)
		rveczzp2.set_color(GREEN)

		rvecxxn2 = Vector([1,0,0])
		rvecxyn2 = Vector([0,1,0])
		rvecxzn2 = Vector([0,0,1])
		rvecyxn2 = Vector([1,0,0])
		rvecyyn2 = Vector([0,1,0])
		rvecyzn2 = Vector([0,0,1])
		rveczxn2 = Vector([1,0,0])
		rveczyn2 = Vector([0,1,0])
		rveczzn2 = Vector([0,0,1])
		rvecxxn2.shift(LEFT*3)
		rvecxyn2.shift(LEFT+DOWN)
		rvecxzn2.shift(LEFT+IN)
		rvecyxn2.shift(DOWN+LEFT)
		rvecyyn2.shift(DOWN*3)
		rvecyzn2.shift(DOWN+IN)
		rveczxn2.shift(IN+LEFT)
		rveczyn2.shift(IN+DOWN)
		rveczzn2.shift(IN*3)
		rvecxxn2.set_color(RED)
		rvecxyn2.set_color(YELLOW)
		rvecxzn2.set_color(GREEN)
		rvecyxn2.set_color(RED)
		rvecyyn2.set_color(YELLOW)
		rvecyzn2.set_color(GREEN)
		rveczxn2.set_color(RED)
		rveczyn2.set_color(YELLOW)
		rveczzn2.set_color(GREEN)


		self.set_camera_orientation(phi=75 * DEGREES,theta= 120*DEGREES)
		self.begin_ambient_camera_rotation(rate=0)

		#self.add(axes)
		self.add_fixed_in_frame_mobjects(title)
		self.add(prism1)
		# self.add(vecxx,vecxy,vecxz,vecyx,vecyy,vecyz,veczx,veczy,veczz)
		self.wait()

		# self.play(
		# 	ApplyMethod(prism1.apply_matrix, matrixrot),
		# 	# ReplacementTransform(vecxxp1,vecxxp2),
		# 	# ReplacementTransform(vecxxn1,vecxxn2)
		#  )
		self.play(Rotate(prism1, 90*DEGREES),run_time=5)
		# self.remove(vecxxp2,vecxxn2)

		self.wait()


class RotationTensorCurl(Scene):
	def construct(self):
		matrix = Matrix(
			[["0", "{1 \\over 2}(\\pdv{v}{x}-\\pdv{u}{y})", "{1 \\over 2}(\\pdv{w}{x}-\\pdv{u}{z})"], 
			["{1 \\over 2}(\\pdv{u}{y}-\\pdv{v}{x})", "0", "{1 \\over 2}(\\pdv{w}{y}-\\pdv{v}{z})"],
			["{1 \\over 2}(\\pdv{u}{z}-\\pdv{w}{x})", "{1 \\over 2}(\\pdv{v}{z}-\\pdv{w}{y})", "0"]
			], v_buff=1.5, h_buff=3.5)

		text1 = TextMobject("The ",
			"rotation or spin tensor")
		text2 = TextMobject("shows ", "curl")
		text = TexMobject("\\underline{\\underline{\\Omega}} ",
			"=")

		text2[1].set_color(PURPLE)
		text1[1].set_color(BLUE)
		text[0].set_color(BLUE)
		matrix[1].set_color(BLUE)
		matrix[2].set_color(BLUE)

		text1.to_edge(UP)
		text2.next_to(text1, DOWN)
		text.shift(LEFT*5)
		matrix.next_to(text)

		Curl = Matrix(["\\pdv{v}{x}-\\pdv{u}{y}","\\pdv{w}{y}-\\pdv{v}{z}","\\pdv{u}{z}-\\pdv{w}{x}"], v_buff=1.5)
		Curl[0][0].set_color(BLUE)
		Curl[0][1].set_color(BLUE)
		Curl[0][2].set_color(BLUE)
		Curl[1].set_color(PURPLE)
		Curl[2].set_color(PURPLE)

		textDiv = TexMobject("= ", "\\curl{","\\textbf{v}}")
		textDiv[1].set_color(PURPLE)
		Curl.shift(DOWN)
		textDiv.next_to(Curl, RIGHT)

		text3b1b = TextMobject("3Blue",
			"1Brown")
		text3b1b[0].set_color(BLUE)
		text3b1b[1].set_color("#964b00")


		self.play(Write(text1))
		self.play(Write(text2))
		self.play(Write(text))
		self.play(Write(matrix))
		self.wait(2)


		self.remove(text, matrix)
		self.play(
			ReplacementTransform(matrix[0][1],Curl[0]),
			ReplacementTransform(matrix[0][5],Curl[1]),
			ReplacementTransform(matrix[0][6],Curl[2])
		)
		self.wait()

		self.play(Write(textDiv))
		self.wait()

		self.play(
			FadeOut(Curl), FadeOut(text1), FadeOut(text2),
			ReplacementTransform(textDiv,text3b1b)
		)
		self.wait()


class FieldCurlExplanationScene(Scene):
	CONFIG={
		"dot_kwargs":{
			"radius":.05,
			"color":YELLOW,
			"fill_opacity": 1,
		}
	}

	def show_rotation(self):
		counterclockwise_arrows = [
				self.get_rotation_arrows(clockwise=False).move_to([0,0,0])
		]
		
		for arrows in counterclockwise_arrows:
			always_rotate(arrows, rate = 30 * DEGREES)
			self.play(
				FadeIn(arrows)
			)
		self.wait()

	def get_rotation_arrows(self, clockwise=True, width=1):
		result = VGroup(*[
			Arrow(
				*points,
				buff=2 * SMALL_BUFF,
				path_arc=90 * DEGREES
			).set_stroke(width=5)
			for points in adjacent_pairs(compass_directions(4, RIGHT))
		])
		if clockwise:
			result.flip()
		result.set_width(width)
		return result


	def construct(self):
		FieldCurl = VectorField(FuncCurl,y_max=int(np.ceil(FRAME_HEIGHT / 2 + 3.5)),y_min=int(np.ceil(FRAME_HEIGHT / 2 - 14)))

		dots=VGroup()
		for vector in FieldCurl:
			dot = Dot(**self.dot_kwargs)
			dot.set_color(WHITE)
			dot.move_to(vector.get_start())
			dot.target = vector
			dots.add(dot)

		title = TextMobject("Curl")
		title.add_background_rectangle()
		title.to_edge(UP)
		
		self.add(FieldCurl)
		self.play(Write(title))
		self.wait()

		# self.play(
		# 	ShowCreation(dots),
		# 	)
		# self.wait()

		# move_submobjects_along_vector_field(dots, FuncCurl)
		# self.wait()

		self.show_rotation()
		self.wait(2)


class VelocityGradientTensorEnd(Scene):
	def construct(self):
		matrix = Matrix(
			[["\\pdv{u}{x}", "\\pdv{v}{x}", "\\pdv{w}{x}"], 
			["\\pdv{u}{y}", "\\pdv{v}{y}", "\\pdv{w}{y}"],
			["\\pdv{u}{z}", "\\pdv{v}{z}", "\\pdv{w}{z}"]
			], v_buff=1.5)

		text1 = TextMobject("Velocity gradient tensor")
		textL = TexMobject("\\underline{\\underline{L}} ",
			"\\equiv (",
			"\\nabla \\textbf{v}",
			")^T =")
		
		text1.set_color(RED)
		textL[0].set_color(RED)
		textL[2].set_color(RED)
		matrix[1].set_color(RED)
		matrix[2].set_color(RED)

		text1.to_edge(UP)
		textL.next_to(matrix, LEFT)

		textM2 = TexMobject("\\pdv{v}{x}")

		self.play(Write(text1))
		self.play(Write(textL))
		self.play(Write(matrix))
		self.wait(3)


class ThankScene(Scene):
	def construct(self):
		thanks = TextMobject("This video was made with Manim, ")
		thanks2 = TextMobject("a python library made by")
		thanks3 = TextMobject("Grant Sanderson from ", "3Blue","1Brown", ".")

		thanks3[1].set_color(BLUE)
		thanks3[2].set_color("#964b00")

		thanks.next_to(thanks2, UP)
		thanks3.next_to(thanks2, DOWN)

		self.play(Write(thanks))
		self.play(Write(thanks2))
		self.play(Write(thanks3))
		self.wait(2)


class EndScene(Scene):
	def construct(self):
		text1 = TextMobject("A video by")
		text2 = TextMobject("Dennis Langelaan")

		text1.next_to(text2, UP)

		self.play(Write(text1))
		self.play(Write(text2))
		self.wait(3)






# class FluidElement(LinearTransformationScene):
# 	CONFIG = {
# 		"include_background_plane": False,
# 		"include_foreground_plane": False,
# 		"foreground_plane_kwargs": {
# 			"x_radius": FRAME_WIDTH,
# 			"y_radius": FRAME_HEIGHT,
# 			"secondary_line_ratio": 0
# 		},
# 		"background_plane_kwargs": {
# 			"color": GREY,
# 			"secondary_color": DARK_GREY,
# 			"axes_color": GREY,
# 			"stroke_width": 2,
# 		},
# 		"show_coordinates": False,
# 		"show_basis_vectors": False,
# 		"basis_vector_stroke_width": 6,
# 		"i_hat_color": X_COLOR,
# 		"j_hat_color": Y_COLOR,
# 		"leave_ghost_vectors": False,
# 	}

# 	def formulas(self):
# 		text1 = TexMobject("\\textbf{v}(\\textbf{x}+d\\textbf{x}) = ",
# 			"\\textbf{v}(\\textbf{x}) + ", 
# 			"\\pdv{\\textbf{v}(\\textbf{x})}{x}",
# 			"dx + ",
# 			"\\pdv{\\textbf{v}(\\textbf{x})}{y}", 
# 			"dy + ",
# 			"\\pdv{\\textbf{v}(\\textbf{x})}{z}",
# 			"dz")

# 		text1[2].set_color(RED)
# 		text1[4].set_color(RED)
# 		text1[6].set_color(RED)

# 		text2 = TexMobject("\\textbf{v}(\\textbf{x}+d\\textbf{x}) = \\textbf{v}(\\textbf{x}) + d\\textbf{x} \\cdot",
# 		 "\\nabla \\textbf{v}(\\textbf{x}) ")

# 		text2[1].set_color(RED)

# 		text1.shift(2*DOWN)
# 		text2.next_to(text1, DOWN)
# 		text2.shift(2.4*LEFT)


# 		self.play(Write(text1))
# 		self.wait(7)
# 		self.play(Write(text2))

		


# 	def construct(self):
# 		matrix = [[np.cos(-1 * TAU/16), np.cos(TAU/4-1 * TAU/16)], [np.sin(-1 * TAU / 16), np.sin(TAU/4-1 * TAU/16)]]

# 		rect = Rectangle(width=4,height=2)
# 		rect.move_to(RIGHT*2+UP*3)

		

# 		labelA = TextMobject("A")
# 		labelA.move_to(RIGHT*.5 + UP*2)
# 		labelB = TextMobject("B")
# 		labelB.move_to(RIGHT*5.5 + UP*2)


# 		v1 = self.add_vector([0,2])
# 		v2 = self.add_vector([4,4])

# 		self.add_transformable_mobject(rect)


# 		self.apply_matrix(matrix)


# 		l1 = self.label_vector(v1, "x")
# 		self.play(Write(labelA))
# 		self.wait(2)

# 		l2 = self.label_vector(v2, "\\vec{\\textbf{x}}+d\\vec{\\textbf{x}}")
# 		self.play(Write(labelB))

# 		self.wait(4)

# 		v3 = self.add_vector([np.cos(PI/4),np.sin(PI/4)])
# 		v4 = self.add_vector([0,1.5])

# 		self.play(ApplyMethod(v3.shift, v1.get_end()))
# 		self.play(ApplyMethod(v4.shift, v2.get_end()))


# 		l3 = TexMobject("\\vec{\\textbf{v}}(\\vec{\\textbf{x}})")
# 		l3.move_to(RIGHT*1.7 + UP*2.8)
# 		l3.scale(0.8)
# 		l3.set_color(YELLOW)

# 		l4 = TexMobject("\\vec{\\textbf{v}}(\\vec{\\textbf{x}}+d\\vec{\\textbf{x}})")
# 		l4.move_to(RIGHT*6.1 + UP*2.8)
# 		l4.scale(0.8)
# 		l4.set_color(YELLOW)

		
# 		self.play(Write(l3))
# 		self.play(Write(l4))

# 		self.wait(4)

# 		self.formulas()

# 		self.wait(3)


# class Translation(LinearTransformationScene):
# 	CONFIG = {
# 		"include_background_plane": False,
# 		"include_foreground_plane": False,
# 		"foreground_plane_kwargs": {
# 			"x_radius": FRAME_WIDTH,
# 			"y_radius": FRAME_HEIGHT,
# 			"secondary_line_ratio": 0
# 		},
# 		"background_plane_kwargs": {
# 			"color": GREY,
# 			"secondary_color": DARK_GREY,
# 			"axes_color": GREY,
# 			"stroke_width": 2,
# 		},
# 		"show_coordinates": False,
# 		"show_basis_vectors": False,
# 		"basis_vector_stroke_width": 6,
# 		"i_hat_color": X_COLOR,
# 		"j_hat_color": Y_COLOR,
# 		"leave_ghost_vectors": False,
# 	}
# 	def construct(self):
# 		rect = Rectangle(height=2, width=4)
# 		vector_array = np.array([[1], [2]])
# 		rect.shift(LEFT)
# 		text = TextMobject("Translation")
# 		text.to_edge(UP)

# 		self.play(Write(text))

# 		self.add_transformable_mobject(rect)



# 		self.play(ApplyMethod(rect.shift, RIGHT*2))

# 		self.wait(3)


# class FieldZeroScene(Scene):
# 	CONFIG={
# 		"dot_kwargs":{
# 			"radius":.05,
# 			"color":YELLOW,
# 			"fill_opacity": 1,
# 		}
# 	}


# 	def construct(self):
# 		FieldZero = VectorField(FuncZero)

# 		rect = Rectangle(height=2, width=4)
# 		rect.shift(LEFT*2)
		
# 		corner1 = Dot([0,1,0], color=WHITE)
# 		corner2 = Dot([0,-1,0], color=WHITE)
# 		corner3 = Dot([-4,-1,0], color=WHITE)
# 		corner4 = Dot([-4,1,0], color=WHITE)
# 		leftdot = Dot([-4,0,0], color=RED)
# 		rightdot = Dot([0,0,0], color=BLUE)


# 		corners=VGroup()
# 		for vector in FieldZero:
# 			dot = Dot(**self.dot_kwargs)
# 			dot.move_to(vector.get_start())
# 			dot.target = vector
# 			corners.add(dot)


# 		self.add(FieldZero)
# 		self.wait()
# 		self.add(rect)
# 		self.add(FieldZero,corner1,corner2,corner3,corner4,leftdot,rightdot)

# 		self.wait()
# 		for dot in corner1,corner2,corner3,corner4,leftdot,rightdot:
# 			move_submobjects_along_vector_field(
# 				dot,
# 				lambda p: FuncZero(p)
# 			)
# 		move_submobjects_along_vector_field(rect, FuncZero)
# 		self.wait(5)

# 		for dot in corner1,corner2,corner3,corner4,leftdot,rightdot:
# 			dot.clear_updaters()
# 		rect.clear_updaters()
		



class TryOut(Scene):
	def construct(self):
		text = TexMobject("test")

		self.add(text)

		self.wait()