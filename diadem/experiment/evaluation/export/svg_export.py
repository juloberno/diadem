# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import svgutils.transform as sg

#text_car = </svg:g><svg:g transform="translate(200, 200) scale(1)  rotate(90.000000 0.000000 0.000000)"><svg:metadata id="metadata8"><rdf:RDF><cc:Work rdf:about=""><dc:format>image/svg+xml
#     </dc:format><dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/><dc:title/></cc:Work></rdf:RDF></svg:metadata><svg:defs id="defs6"><svg:clipPath clipPathUnits="userSpaceOnUse" id="clipPath20">
#<svg:path d="M 8000,0 H 0 V 6000 H 8000 V 0 M 4000,1098.33 c -1250.17,0 -2263.63,-171.189 -2263.63,-382.35 0,-211.171 1013.46,-382.359 2263.63,-382.359 1250.17,0 2263.63,171.188 2263.63,382.359 0,211.161 -1013.46,382.35 -2263.63,382.35" id="path18" inkscape:connector-curvature="0"/>
#</svg:clipPath><svg:radialGradient fx="0" fy="0" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="matrix(7056.18,0,0,-7056.18,4000,3000)" spreadMethod="pad" id="radialGradient28"><svg:stop style="stop-opacity:1;stop-color:#ffffff" offset="0" id="stop22"/>
#<svg:stop style="stop-opacity:1;stop-color:#ffffff" offset="0.78266539" id="stop24"/><svg:stop style="stop-opacity:1;stop-color:#231f20" offset="1" id="stop26"/></svg:radialGradient><svg:clipPath clipPathUnits="userSpaceOnUse" id="clipPath38">
#<svg:path d="m 4000,333.621 c -1250.17,0 -2263.63,171.188 -2263.63,382.359 0,211.161 1013.46,382.35 2263.63,382.35 1250.17,0 2263.63,-171.189 2263.63,-382.35 0,-211.171 -1013.46,-382.359 -2263.63,-382.359" id="path36" inkscape:connector-curvature="0"/>
#</svg:clipPath></svg:defs><sodipodi:namedview pagecolor="#ffffff" bordercolor="#666666" borderopacity="1" objecttolerance="10" gridtolerance="10" guidetolerance="10" inkscape:pageopacity="0" inkscape:pageshadow="2" inkscape:window-width="2560" inkscape:window-height="1361" id="namedview4" showgrid="false" inkscape:zoom="0.59" inkscape:cx="81.815374" inkscape:cy="322.23958" inkscape:window-x="-9" inkscape:window-y="-9" inkscape:window-maximized="1" inkscape:current-layer="g12" fit-margin-top="0" fit-margin-left="0" fit-margin-right="0" fit-margin-bottom="0"/><svg:g id="g10" inkscape:groupmode="layer" inkscape:label="ink_ext_XXXXXX" transform="matrix(1.3333333,0,0,-1.3333333,-407.45033,669.15998)"><svg:g id="g12" transform="scale(0.1)"><svg:g id="g14"/><svg:path d="m 4766.63,2207.48 c -0.1,-9.98 -6.36,-18.9 -15.78,-22.21 -8.85,-3.11 -17.81,-6.15 -26.88,-9.14 -13.81,-4.55 -28.48,3.98 -31.03,18.29 -8.47,47.44 -25.15,156.93 -32.39,317 -6.55,144 -7.39,336.57 -7.39,336.57 v 128.47 c 0,0.28 16.28,233.92 26.76,308.77 9.51,67.17 31.86,184.25 53.32,291.44 5.67,28.34 47.43,23.97 47.13,-4.92 z m -101.78,-690.5 c -191.98,-98.86 -424.46,-151.15 -659.38,-157.01 -234.91,5.86 -467.39,58.15 -659.38,157.01 -26.76,13.94 -46.03,52.11 -41.07,78.31 24.14,126.88 48.93,258.72 74.48,394.58 3.93,21.34 21.89,32.28 43.32,26.48 174.83,-47.9 376.33,-73.55 579.78,-77.03 h 5.74 c 203.46,3.48 404.95,29.13 579.79,77.03 21.42,5.8 39.39,-5.14 43.32,-26.48 25.54,-135.86 50.34,-267.7 74.48,-394.58 4.96,-26.2 -14.32,-64.37 -41.08,-78.31 z m -1377.87,659.15 c -9.07,2.99 -18.03,6.03 -26.88,9.14 -9.42,3.31 -15.69,12.23 -15.79,22.21 l -13.74,1364.27 c -0.29,28.89 41.46,33.26 47.13,4.92 21.46,-107.19 43.82,-224.27 53.33,-291.44 10.48,-74.85 26.76,-308.49 26.76,-308.77 v -128.47 c 0,0 -0.84,-192.57 -7.39,-336.57 -7.25,-160.07 -23.93,-269.56 -32.39,-317 -2.56,-14.31 -17.23,-22.84 -31.03,-18.29 z m -50.9,2265.25 c 11.64,34.9 19.78,143.47 69.4,250.09 71.85,154.38 144.54,189.68 175.09,197.75 9.24,2.44 18.99,-0.72 24.86,-8.26 24.31,-31.26 71.98,-131.42 -116,-302.29 l -1.59,-1.33 c -78.09,-90.24 -65.25,-127.3 -144.23,-144.71 -5.04,-1.11 -9.17,3.85 -7.53,8.75 z m 772.26,-430.6 c 238.26,-4.19 474,-39.86 667.18,-107.16 18.81,-6.64 32.65,-30.69 29.57,-50.15 -26.67,-167.49 -54.18,-341.15 -82.44,-519.96 -2.62,-16.37 -15.72,-27.23 -31.34,-25.64 -175.52,18.1 -378.28,27.76 -582.97,29.04 v 0.04 c -0.96,0 -1.91,-0.01 -2.87,-0.02 -0.95,0.01 -1.91,0.02 -2.87,0.02 v -0.04 c -204.68,-1.28 -407.45,-10.94 -582.96,-29.04 -15.63,-1.59 -28.73,9.27 -31.35,25.64 -28.25,178.81 -55.76,352.47 -82.43,519.96 -3.09,19.46 10.76,43.51 29.57,50.15 193.17,67.3 428.91,102.97 667.17,107.16 z m 497.18,870.18 c 5.86,7.54 15.62,10.7 24.85,8.26 30.56,-8.07 103.24,-43.37 175.09,-197.75 49.62,-106.62 57.76,-215.19 69.41,-250.09 1.63,-4.9 -2.5,-9.86 -7.54,-8.75 -78.97,17.41 -66.13,54.47 -144.22,144.71 l -1.6,1.33 c -187.98,170.87 -140.3,271.03 -115.99,302.29 z m 320.65,-1042.12 c 4.43,310.49 8.88,511.26 8.88,514.01 -0.09,164.77 -50.34,495.16 -240.19,571.42 -136.97,55.01 -359.24,88.23 -586.52,94.25 v 0.18 c -0.96,-0.02 -1.91,-0.07 -2.87,-0.09 -0.95,0.02 -1.91,0.07 -2.87,0.09 v -0.18 c -227.28,-6.02 -449.54,-39.24 -586.52,-94.25 -189.85,-76.26 -240.09,-406.65 -240.19,-571.42 0,-2.71 4.34,-198.26 8.71,-501.82 -12.1,-13.33 -200.5,-222.96 -98.62,-248.07 50.2,-12.38 82.07,-1.4 101.7,13.46 1.17,-98.62 2.27,-204.17 3.18,-314.81 5.06,-597.25 5.06,-1343.83 -14.97,-1867.07 -3.46,-146.52 89.55,-338.43 288.85,-394.39 157.56,-44.271 357.01,-56.279 537.86,-56.349 v -2.5 c 0.95,0.008 1.91,0.008 2.87,0.019 0.96,-0.011 1.92,-0.011 2.87,-0.019 v 2.5 c 180.86,0.07 380.3,12.078 537.87,56.349 199.3,55.96 292.3,247.87 288.84,394.39 -20.02,523.24 -20.02,1269.82 -14.97,1867.07 0.9,108.03 1.96,211.19 3.1,307.8 19.91,-10.85 49,-16.76 90.83,-6.45 89.98,22.18 -46.41,188.21 -87.84,235.88"
#       style="fill:#3a6289;fill-opacity:1;fill-rule:nonzero;stroke:none" id="path82" inkscape:connector-curvature="0"/></svg:g></svg:g></svg:g></svg:g>

class SvgScenarioRenderer:
    def __init__(self):
        self.main_svg = None
        self.car_svg = None
        self.car_cent_x = None
        self.car_cent_y = None
        self.car_cent_theta = None
        self.x_ccord_offset = 0.0
        self.y_coord_offset = 0.0
        self.theta_coord_offset = 0.0
        self.main_axis_name = "axis_1"


    def setMainSVG(self, **kwargs):
        if "filename" in kwargs:
            self.main_svg = sg.fromfile(kwargs["filename"])
        elif "mpl_fig" in kwargs:
            self.main_svg = sg.from_mpl(kwargs["mpl_fig"])


    def setCarSVG(self,filename):
        self.car_svg = sg.fromfile(filename)

    def replace(self, x,y, theta, scale ):
        main_root = self.main_svg.getroot()
        car_obj = main_root.find_id("patch_8")
        # todo: iterate over all car patches -> draw in scenario then read x/y from rectangle and scale vehicle accordingly
        car_root = self.car_svg.getroot()
        car_root.rotate(theta)
        car_root.moveto(x,y,scale = scale)
        self.main_svg.append([main_root, car_root])

    def export(self, filename):
        self.main_svg.save(filename)

if __name__ == "__main__":
    renderer = SvgScenarioRenderer()

    renderer.setMainSVG(filename="sample_11.svg")
    renderer.setCarSVG("car_blue.svg")
    renderer.replace(200,200,90,1)
    renderer.export("test.svg")
