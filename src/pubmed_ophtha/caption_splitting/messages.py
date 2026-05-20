"""Module defining system and few-shot messages for caption splitting functionality."""

from collections.abc import Sequence
from typing import Any, Type

import pydantic

from pubmed_ophtha.caption_splitting.response_models import (
    SplitSubCaptions,
    SubCaptionNames,
)

NAMING_SYSTEM_MESSAGE_QWEN = """\
Your job is to extract the names of subfigures from captions. \
The names are usually single letters (like A, B, C) or words (like Left, Right, Top, \
Bottom). If there are no names in the caption, return an empty list. \
The names can be one of the types:
- Either single letters (A, B, C, etc.) or letters with parentheses ((A), (B), (C), etc\
.) and other formatting.
- Or positional words like Left, Right, Top, Bottom, Center, Top left, Bottom right, \
etc.
- Or column indicators like First column, Second column, Middle column, Last column, \
etc.
- Or row indicators like First row, Second row, Middle row, Last row, etc.
"""

SPLITTING_SYSTEM_MESSAGE_QWEN = """\
Your job is to split captions into subcaptions for each figure. \
Do not add additional text that is not in the caption. In total the subcaptions \
should contain all of the text from the caption. Format your response in JSON. \
The keys are the subfigure labels (a, b, c, etc.) and the values are the corresponding \
subcaptions. Use the provided subfigure names to guide your splitting.

Subtasks:
    - Detect the number of subfigures in the caption.
    - Find the names of the subfigures.
    - Split the caption into subcaptions for each subfigure.
    - Construct a JSON response with the subcaptions.

Potential errors:
    - The caption may not explicitly mention all subfigures. Sometimes the names are \
        missing but can be deducted from context or the subcaption names are \
        summarized (e.g. A-C means there are subcaptions A, B, and C).
    - Some figures do not have subcaptions, in which case the caption should be \
        returned under the key 'None'.
    - The key 'None' should not be included if the caption describes multiple \
        subfigures.
    - Sometimes only a part of the caption is relevant to a subfigure, \
        so you should split the caption accordingly.
    - Sometimes a sentence in a caption is relevant to multiple subfigures, \
        so you should split the caption accordingly.
    - Do not include any additional text that is not part of the caption. \
        The response should only contain the subcaptions in JSON format.
    - Do not include the subfigure name in the subcaption text if not explicitly \
        necessary to understand the subcaption. For example, if the caption is \
        "A: This is figure A. B: This is figure B.", then the subcaption for A \
        should be "This is figure A." and not "A: This is figure A.". Another example: \
        if the caption is "A depth map (A) and a segmentation map (B) of the Mariana \
        trench.", then the subcaption for A should be "A depth map of the Mariana \
        trench." and not "A depth map (A) of the Mariana trench.". If the subcaption \
        contains a reference to another subfigure, then mentioning the name of the \
        other subfigure is necessary to understand the subcaption. For example, if \
        the caption is "A: This is figure A. B: This is figure B, which is related to \
        A.", then the subcaption for B should be "This is figure B, which is related \
        to A.". Another example: if the caption is "A depth map (A) and a segmentation \
        map (B) of the Mariana trench. The segmentation map (B) shows the different \
        geological features.", then the subcaption for B should be "A segmentation map \
        of the Mariana trench. The segmentation map shows the different geological \
        features." and not "A segmentation map of the Mariana trench. The segmentation \
        map (B) shows the different geological features.". Another example: if the \
        caption is "A map of the world (A) and a zoomed-in view of Europe (B). The \
        rectangle in (A) shows the area covered by (B).", then the subcaption for A \
        should be "A map of the world. The rectangle shows the area covered by (B)." \
        and the subcaption for B should be "A zoomed-in view of Europe."
    - Use the correct figure name. If the figure is named (A), then the key should \
        also be '(A)'.

Examples of subcaption titles: 'a', 'b', 'c', 'd', 'Top', 'Left', 'Right', \
'first row', 'last column', 'bottom row', 'center'.
"""

NAMING_FEW_SHOT_EXAMPLES_QWEN: list[dict[str, str | SubCaptionNames]] = [
    {  # PMC2569141_BJ1-92-05-0623-f03
        "user": "(A, B) Fundus photograph and AF image of patient 9 showing a large \
area of central atrophy corresponding to the decreased AF. There is extensive speckled \
AF surrounding the atrophy. This is the only patient who lost fixation due to \
progression of atrophic changes. Patient consent has been obtained for publication of \
this figure.",
        "assistant": SubCaptionNames.model_validate(
            {"names": [{"name": "A"}, {"name": "B"}]}
        ),
    },
    {  # PMC7082944_12886_2020_1382_Fig8_HTML
        "user": "Visualization of feature maps as heat maps. Left images are input \
images. Right images are feature maps superimposed on an input image. Top to bottom, \
panels show CNV, DME, DRUSEN, and NORMAL",
        "assistant": SubCaptionNames.model_validate(
            {
                "names": [
                    {"name": "Bottom"},
                    {"name": "Second"},
                    {"name": "Third"},
                    {"name": "Top"},
                ]
            }
        ),
    },
    {  # PMC6015755_peerj-06-5042-g005
        "user": "Aqz expressed in ectodermally derived tissue.(A) shows an anti-GFP \
staining for GFP::Aqz expression together with an epidermal marker, anti-Cora, showing \
some coincidental expression (yellow). For all immune-stainings in the figure, the \
panels marked underneath the top A–D labeled E–F show the GFP::Aqz channel separate, \
and the G–H show the anti-Aqz. The bottom row of panels marked with I, J, K and L \
depict the anti-coracle (I), anti-Elav (J), anti-22C10 (K), and anti-repo (L) \
channels, respectively. (B) shows anti-GFP for GFP::Aqz and anti-Elav, to mark \
neuronal cells in the developing central NS. (C) Using monoclonal 22C10, there is \
coincidental expression with anti-Aqz antibody, showing that some neurons in the PNS \
are positive for both Aqz and 22C10 (yellow). (D) In contrast no glial cells (marked \
by anti-Repo antibody) are coincidental with anti-Aqz antibody staining, showing \
that glial cells do not express Aqz. Scale bar is 100 µm.",
        "assistant": SubCaptionNames.model_validate(
            {
                "names": [
                    {"name": "A"},
                    {"name": "B"},
                    {"name": "C"},
                    {"name": "D"},
                    {"name": "E"},
                    {"name": "F"},
                    {"name": "G"},
                    {"name": "H"},
                    {"name": "I"},
                    {"name": "J"},
                    {"name": "K"},
                    {"name": "L"},
                ]
            }
        ),
    },
    {  # PMC5592085_peerj-05-3763-g009
        "user": "Images from ONHSD database.(A–C) Examples of correct segmentations. \
(D–F) Examples of incorrect segmentations. The truth OD boundary is marked in green.",
        "assistant": SubCaptionNames.model_validate(
            {
                "names": [
                    {"name": "(a)"},
                    {"name": "(b)"},
                    {"name": "(c)"},
                    {"name": "(d)"},
                    {"name": "(e)"},
                    {"name": "(f)"},
                ]
            }
        ),
    },
    {  # PMC5357819_40942_2017_69_Fig6_HTML
        "user": "Birdshot uveitis. A 58-year-old woman presented with photophobia in \
both eyes. White, deep, multifocal, choroidal lesions were seen on examination, mostly \
located nasally (top row). Fundus autofluorescence showed that these lesions were \
hypoautofluorescent (second row). The lesions were hyperfluorescent on fluorescein \
angiography (third row), and corresponded to hypocyanescent spots on indocyanine \
green (bottom row). Spectral-domain optical coherence tomographies of the fovea in \
both eyes were relatively unremarkable. Enhanced-depth imaging of lesions seen on \
previous imaging (yellow circles) demonstrates focally disrupted outer retinal \
structures with increased transmission into the choroid. Work-up revealed HLA-A29 \
positivity",
        "assistant": SubCaptionNames.model_validate(
            {
                "names": [
                    {"name": "bottom row"},
                    {"name": "second row"},
                    {"name": "third row"},
                    {"name": "top row"},
                ]
            }
        ),
    },
    {  # PMC5052491_JMedLife-09-092-g002
        "user": "Fig. 2L.E. fundus image showing a round area of subfoveal\nfluid",
        "assistant": SubCaptionNames.model_validate({"names": []}),
    },
    {  # PMC8096541_CRIC2021-9985568.002
        "user": "Fundus photograph of the left eye branch retinal artery occlusion \
(BRAO) showing white pallor at the inferior temporal branch of the central retinal \
artery.",
        "assistant": SubCaptionNames.model_validate({"names": []}),
    },
    {  # PMC11268348_gr2
        "user": "Case example of a 21-year-old female patient diagnosed with an AJCC \
T2-staged choroidal melanoma in her right eye. A, B) Ophthalmic images at diagnosis. \
The orange-shaded arrow indicates the lesion. A: Fundus photography. B: B-scan \
ultrasound. Dimensions on ultrasound were of 8.4 mm for largest basal diameter and of \
5.3 mm for tumor apical height. C, D) Axial dose distributions shown on the planning \
CT in respect to the gazing angle hold by the patient during the scan. For proton \
simulation, a straight gazing angle [Ψ = 0 degrees; ɸ=0 degrees] was considered. Ψ \
represented any elevation/depression of the eye, whereas ɸ represented any \
ab-/adduction. E) DVH results for both modalities, stereotactic radiotherapy (SRT) in \
solid line and proton in dashed line. Results are presented for a total treatment \
dose, in RBE-weighted dose (in Gy), or percentage of irradiated volume (%-point). (For \
interpretation of the references to colour in this figure legend, the reader is \
referred to the web version of this article.)",
        "assistant": SubCaptionNames.model_validate(
            {
                "names": [
                    {"name": "A."},
                    {"name": "B."},
                    {"name": "C."},
                    {"name": "D."},
                    {"name": "E."},
                ]
            }
        ),
    },
    {  # PMC9690312_genes-13-02034-g001
        "user": "A peripapillary vessel map of the retinal vessel image of a patient \
with retinitis pigmentosa. Retinal vessel diameters in retinal arterioles (D-A, µm) \
and retinal venules (D-V, µm), were measured within the peripapillary annulus in \
control subjects as well as in RP patients. The corresponding retinal vessel oxygen \
saturation values (A-SO2, V-SO2 and A-V SO2) were calculated, as well. The colors in \
the oximetry image indicate the relative oxygen saturation (%) in retinal vessels as \
quantified in the scale bar. Note the increased A-SO2 compared to V-SO2.",
        "assistant": SubCaptionNames.model_validate({"names": []}),
    },
    {  # PMC3938601_pone.0090129.g005
        "user": "Optic disc photograph (A), en-face image (B), B-scan SD-OCT images \
(a, b, c), and infrared (IR) image (C) with circumpapillary B-scan image (D) in case \
22.(A, B) A pit is seen near the disc margin (black arrows). (a, b, c) Horizontal \
B-scan images obtained at the location of pit (marked with dotted lines in A and B). \
Note the LC defect (large white arrows) at the location of optic disc pit. \
Retinoschisis is observed in all three B-scans (large arrow heads). (C, D) \
Retinoschisis is seen adjacent to the optic disc pit in both the IR (small white \
arrow) and the circumpapillary B-scan image (small arrow heads).",
        "assistant": SubCaptionNames.model_validate(
            {
                "names": [
                    {"name": "A"},
                    {"name": "B"},
                    {"name": "C"},
                    {"name": "D"},
                    {"name": "a"},
                    {"name": "b"},
                    {"name": "c"},
                ]
            }
        ),
    },
    {  # PMC9719031_cureus-0014-00000031040-i02
        "user": "Fluorescein angiogram of the right and left eye.Fluorescein angiogram \
showing a choroidal neovascular membrane adjacent to the disc. The picture on the left \
side is for the right eye and the one on the right side is for the left eye.",
        "assistant": SubCaptionNames.model_validate(
            {"names": [{"name": "left"}, {"name": "right"}]}
        ),
    },
    {  # PMC8222724_fphar-12-632295-g002
        "user": "Presentation of the rat retina before (H) and after retrobulbar \
administered agents (saline (S), BPC 157 (B), and L-NAME (N)), immediately (I), and at \
20 min (II) (IS, IB, IN; IIS, IIB, and IIN). H, before agent’s application \
(H—healthy). Normal eye background, normal presentation of the retinal and choroidal \
blood vessels (A). I, immediately after agent’s application (IS, IB, and IN) (B). \
Normal eye background, normal presentation of the retinal and choroidal blood vessels \
(saline (IS), BPC 157 (IB); moderate generalized irregularity in diameter of blood \
vessels with moderate atrophy of the optic disk, faint presentation of the choroidal \
blood vessels (IN). II, 20 min after agent’s application (IIS, IIB, and IIN) (C). \
Normal eye background, normal presentation of the retinal and choroidal blood vessels \
(saline (IIS), BPC 157 (IIB) (same presentation maintained till the end of the \
experiment, data not specifically shown); moderate generalized irregularity diameter \
blood vessels with moderate atrophy of the optic disk, faint presentation of the \
choroidal blood vessels (IIN). The images are processed with software purchased with a \
USB microscope camera “Veho Discovery VMS-004 Deluxe.”",
        "assistant": SubCaptionNames.model_validate(
            {"names": [{"name": "A"}, {"name": "B"}, {"name": "C"}]}
        ),
    },
    {  # PMC7878975_gr2
        "user": "Response of left eye to treatment.2a. Day 1 (Top)Left eye had less \
numbers of cotton wool spot and an incomplete macular star. (Middle) Central serous \
retinal detachment was with increased choroidal thickness. (Lower and bottom) OCTA and \
vascular density map of choriocapillaris as heat map showing coarse pattern. 2b. Day \
10 (Top) Fundus photos showing decreased areas of cotton wool spots.(Middle) Decreased \
sub-retinal fluid level but choroidal thickness increased. (lower and Bottom) \
Choriocapillaris signals and vascular density map returning to more fine signals but \
still with presence of flow voids. 2c. (Day 24)(Top) Normal fundus with sharp foveolar \
reflex. (Middle) Complete resolution of subretinal fluid with decreased macular \
thickness with normalization of choroidal thickness. (Lower and Bottom) OCTA and \
vascular density map of the choroid showing near normal choriocapillaris pattern.",
        "assistant": SubCaptionNames.model_validate(
            {"names": [{"name": "2a."}, {"name": "2b."}, {"name": "2c."}]}
        ),
    },
    {  # PMC6189960_micromachines-07-00090-g010
        "user": "Illustration of the image analysis steps. (a) Phase contrast \
microscope image of a single cell. The center point (red color) of the cell is \
manually selected and the circular border (light blue color) for the polar \
transformation is determined by the original rectangular image border; (b) The \
polar-transformed image; (c) shows a plot of the gray-scale intensity along the green \
vertical line appearing in (b). The raw data (red line) is smoothed by Fourier \
filtering (blue line); (d) shows the first derivative of the blue-line (intensity) and \
the red circle shows the point identified as minimum, identified as belonging to cell \
border; (e) by repeating the same procedure for all the polar angles, the cell border \
is reconstructed on the polar image and (f) then transformed back to original image.",
        "assistant": SubCaptionNames.model_validate(
            {
                "names": [
                    {"name": "a)"},
                    {"name": "b)"},
                    {"name": "c)"},
                    {"name": "d)"},
                    {"name": "e)"},
                    {"name": "f)"},
                ]
            }
        ),
    },
    {  # PMC8072621_ijms-22-04160-g001
        "user": "Cuticle on epidermis of arabidopsis sepal (A,B) and hypocotyl (C,D). \
Cross-sections of outer periclinal walls are shown in micrographs obtained using \
transmission electron microscopy. (A,B) Outer periclinal wall of the cell of the \
abaxial sepal epidermis. The cell wall layers labelled in (B) are: epicuticular waxes \
(EW); cuticle proper (CP); cuticular layer (CL); and non-cutinized primary wall (PW). \
The cuticle proper is folded. (C,D) Outer periclinal wall of an epidermal cell of the \
etiolated hypocotyl. The cuticle (Cu) overlying coves of the primary cell wall (PW) is \
very thin and its layers cannot be distinguished.",
        "assistant": SubCaptionNames.model_validate(
            {"names": [{"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"}]}
        ),
    },
    {  # PMC7826764_jcm-10-00232-g001
        "user": "Rate of preserved autofluorescence (PAF) area loss seen in fundus \
autofluorescence (FAF) images. Measurement of the PAF area at baseline (a). The foveal \
area remains intact with consistent PAF, surrounded by an area of mottled AF. The \
second follow-up timepoint is from four years later (b). The baseline PAF area is \
shown as the solid line, while follow up PAF is a dashed line (c). Alignment of the \
patient to camera while imaging can cause differences in illumination across images (a \
to b), which may lead to differences in measurement sizes.",
        "assistant": SubCaptionNames.model_validate(
            {"names": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}
        ),
    },
    {  # PMC3991579_pone.0093624.g002
        "user": "Preprocessing: (from left to right) Original image (A), field of view \
mask (B) and enhanced image after uneven illumination correction (C).",
        "assistant": SubCaptionNames.model_validate(
            {"names": [{"name": "A"}, {"name": "B"}, {"name": "C"}]}
        ),
    },
]


SPLITTING_FEW_SHOT_EXAMPLES_QWEN: list[dict[str, str | SplitSubCaptions]] = [
    {  # PMC7082944_12886_2020_1382_Fig8_HTML
        "user": 'Visualization of feature maps as heat maps. Left images are input \
images. Right images are feature maps superimposed on an input image. Top to bottom, \
panels show CNV, DME, DRUSEN, and NORMAL\n\nThe predicted subfigure names are: \
{"names":[{"name":"Bottom"},{"name":"Second"},{"name":"Third"},{"name":"Top"}]}.\n',
        "assistant": SplitSubCaptions.model_validate(
            {
                "sub_captions": {
                    "Bottom": {
                        "text": "Visualization of feature maps as heat maps. Left \
images are input images. Right images are feature maps superimposed on an input image. \
Panel shows NORMAL."
                    },
                    "Second": {
                        "text": "Visualization of feature maps as heat maps. Left \
images are input images. Right images are feature maps superimposed on an input image. \
Panel shows DME."
                    },
                    "Third": {
                        "text": "Visualization of feature maps as heat maps. Left \
images are input images. Right images are feature maps superimposed on an input image. \
Panel shows DRUSEN."
                    },
                    "Top": {
                        "text": "Visualization of feature maps as heat maps. Left \
images are input images. Right images are feature maps superimposed on an input image. \
Panel shows CNV."
                    },
                }
            }
        ),
    },
    {  # PMC9719031_cureus-0014-00000031040-i02
        "user": 'Fluorescein angiogram of the right and left eye.Fluorescein angiogram \
showing a choroidal neovascular membrane adjacent to the disc. The picture on the left \
side is for the right eye and the one on the right side is for the left eye.\n\nThe \
predicted subfigure names are: {"names":[{"name":"left"},{"name":"right"}]}.\n',
        "assistant": SplitSubCaptions.model_validate(
            {
                "sub_captions": {
                    "left": {
                        "text": "Fluorescein angiogram of the left eye. Fluorescein \
angiogram showing a choroidal neovascular membrane adjacent to the disc."
                    },
                    "right": {
                        "text": "Fluorescein angiogram of the right eye. Fluorescein \
angiogram showing a choroidal neovascular membrane adjacent to the disc."
                    },
                }
            }
        ),
    },
    {  # PMC8222724_fphar-12-632295-g002
        "user": 'Presentation of the rat retina before (H) and after retrobulbar \
administered agents (saline (S), BPC 157 (B), and L-NAME (N)), immediately (I), and at \
20 min (II) (IS, IB, IN; IIS, IIB, and IIN). H, before agent’s application \
(H—healthy). Normal eye background, normal presentation of the retinal and choroidal \
blood vessels (A). I, immediately after agent’s application (IS, IB, and IN) (B). \
Normal eye background, normal presentation of the retinal and choroidal blood vessels \
(saline (IS), BPC 157 (IB); moderate generalized irregularity in diameter of blood \
vessels with moderate atrophy of the optic disk, faint presentation of the choroidal \
blood vessels (IN). II, 20 min after agent’s application (IIS, IIB, and IIN) (C). \
Normal eye background, normal presentation of the retinal and choroidal blood vessels \
(saline (IIS), BPC 157 (IIB) (same presentation maintained till the end of the \
experiment, data not specifically shown); moderate generalized irregularity diameter \
blood vessels with moderate atrophy of the optic disk, faint presentation of the \
choroidal blood vessels (IIN). The images are processed with software purchased with a \
USB microscope camera “Veho Discovery VMS-004 Deluxe.”\n\nThe predicted subfigure \
names are: {"names":[{"name":"A"},{"name":"B"},{"name":"C"}]}.\n',
        "assistant": SplitSubCaptions.model_validate(
            {
                "sub_captions": {
                    "A": {
                        "text": "Presentation of the rat retina before (H) and after \
retrobulbar administered agents (saline (S), BPC 157 (B), and L-NAME (N)), immediately \
(I), and at 20 min (II) (IS, IB, IN; IIS, IIB, and IIN). H, before agent’s application \
(H—healthy). Normal eye background, normal presentation of the retinal and choroidal \
blood vessels. The images are processed with software purchased with a USB microscope \
camera “Veho Discovery VMS-004 Deluxe.”"
                    },
                    "B": {
                        "text": "Presentation of the rat retina before (H) and after \
retrobulbar administered agents (saline (S), BPC 157 (B), and L-NAME (N)), immediately \
(I), and at 20 min (II) (IS, IB, IN; IIS, IIB, and IIN). H, before agent’s application \
(H—healthy). I, immediately after agent’s application (IS, IB, and IN). Normal eye \
background, normal presentation of the retinal and choroidal blood vessels \
(saline (IS), BPC 157 (IB); moderate generalized irregularity in diameter of blood \
vessels with moderate atrophy of the optic disk, faint presentation of the choroidal \
blood vessels (IN). The images are processed with software purchased with a USB \
microscope camera “Veho Discovery VMS-004 Deluxe.”"
                    },
                    "C": {
                        "text": "Presentation of the rat retina before (H) and after \
retrobulbar administered agents (saline (S), BPC 157 (B), and L-NAME (N)), immediately \
(I), and at 20 min (II) (IS, IB, IN; IIS, IIB, and IIN). H, before agent’s application \
(H—healthy). II, 20 min after agent’s application (IIS, IIB, and IIN). Normal eye \
background, normal presentation of the retinal and choroidal blood vessels \
(saline (IIS), BPC 157 (IIB) (same presentation maintained till the end of the \
experiment, data not specifically shown); moderate generalized irregularity diameter \
blood vessels with moderate atrophy of the optic disk, faint presentation of the \
choroidal blood vessels (IIN). The images are processed with software purchased with a \
USB microscope camera “Veho Discovery VMS-004 Deluxe.”"
                    },
                }
            }
        ),
    },
    {  # PMC6189960_micromachines-07-00090-g010
        "user": 'Illustration of the image analysis steps. (a) Phase contrast \
microscope image of a single cell. The center point (red color) of the cell is \
manually selected and the circular border (light blue color) for the polar \
transformation is determined by the original rectangular image border; (b) The \
polar-transformed image; (c) shows a plot of the gray-scale intensity along the green \
vertical line appearing in (b). The raw data (red line) is smoothed by Fourier \
filtering (blue line); (d) shows the first derivative of the blue-line (intensity) and \
the red circle shows the point identified as minimum, identified as belonging to cell \
border; (e) by repeating the same procedure for all the polar angles, the cell border \
is reconstructed on the polar image and (f) then transformed back to original image.\
\n\nThe predicted subfigure names are: {"names":[{"name":"a)"},{"name":"b)"},\
{"name":"c)"},{"name":"d)"},{"name":"e)"},{"name":"f)"}]}.\n',
        "assistant": SplitSubCaptions.model_validate(
            {
                "sub_captions": {
                    "a)": {
                        "text": "Illustration of the image analysis steps. Phase \
contrast microscope image of a single cell. The center point (red color) of the cell \
is manually selected and the circular border (light blue color) for the polar \
transformation is determined by the original rectangular image border."
                    },
                    "b)": {
                        "text": "Illustration of the image analysis steps. The \
polar-transformed image."
                    },
                    "c)": {
                        "text": "Illustration of the image analysis steps. a plot of \
the gray-scale intensity along the green vertical line appearing in (b). The raw data \
(red line) is smoothed by Fourier filtering (blue line)."
                    },
                    "d)": {
                        "text": "Illustration of the image analysis steps. the first \
derivative of the blue-line (intensity) and the red circle shows the point identified \
as minimum, identified as belonging to cell border."
                    },
                    "e)": {
                        "text": "Illustration of the image analysis steps. By \
repeating the same procedure for all the polar angles, the cell border is \
reconstructed on the polar image."
                    },
                    "f)": {
                        "text": "Illustration of the image analysis steps. By \
repeating the same procedure for all the polar angles, the cell border is \
reconstructed on the polar image and then transformed back to original image."
                    },
                }
            }
        ),
    },
    {  # PMC8072621_ijms-22-04160-g001
        "user": 'Cuticle on epidermis of arabidopsis sepal (A,B) and hypocotyl (C,D). \
Cross-sections of outer periclinal walls are shown in micrographs obtained using \
transmission electron microscopy. (A,B) Outer periclinal wall of the cell of the \
abaxial sepal epidermis. The cell wall layers labelled in (B) are: epicuticular waxes \
(EW); cuticle proper (CP); cuticular layer (CL); and non-cutinized primary wall (PW). \
The cuticle proper is folded. (C,D) Outer periclinal wall of an epidermal cell of the \
etiolated hypocotyl. The cuticle (Cu) overlying coves of the primary cell wall (PW) is \
very thin and its layers cannot be distinguished.\n\nThe predicted subfigure names \
are: {"names":[{"name":"A"},{"name":"B"},{"name":"C"},{"name":"D"}]}.\n',
        "assistant": SplitSubCaptions.model_validate(
            {
                "sub_captions": {
                    "A": {
                        "text": "Cuticle on epidermis of arabidopsis sepal. \
Cross-sections of outer periclinal walls are shown in micrographs obtained using \
transmission electron microscopy. Outer periclinal wall of the cell of the abaxial \
sepal epidermis."
                    },
                    "B": {
                        "text": "Cuticle on epidermis of arabidopsis sepal. \
Cross-sections of outer periclinal walls are shown in micrographs obtained using \
transmission electron microscopy. Outer periclinal wall of the cell of the abaxial \
sepal epidermis. The cell wall layers are: epicuticular waxes (EW); cuticle proper \
(CP); cuticular layer (CL); and non-cutinized primary wall (PW). The cuticle proper is \
folded."
                    },
                    "C": {
                        "text": "Cuticle on epidermis of hypocotyl. Cross-sections of \
outer periclinal walls are shown in micrographs obtained using transmission electron \
microscopy. Outer periclinal wall of an epidermal cell of the etiolated hypocotyl. The \
cuticle (Cu) overlying coves of the primary cell wall (PW) is very thin and its layers \
cannot be distinguished."
                    },
                    "D": {
                        "text": "Cuticle on epidermis of hypocotyl. Cross-sections of \
outer periclinal walls are shown in micrographs obtained using transmission electron \
microscopy. Outer periclinal wall of an epidermal cell of the etiolated hypocotyl. The \
cuticle (Cu) overlying coves of the primary cell wall (PW) is very thin and its layers \
cannot be distinguished."
                    },
                }
            }
        ),
    },
    {  # PMC7826764_jcm-10-00232-g001
        "user": 'Rate of preserved autofluorescence (PAF) area loss seen in fundus \
autofluorescence (FAF) images. Measurement of the PAF area at baseline (a). The foveal \
area remains intact with consistent PAF, surrounded by an area of mottled AF. The \
second follow-up timepoint is from four years later (b). The baseline PAF area is \
shown as the solid line, while follow up PAF is a dashed line (c). Alignment of the \
patient to camera while imaging can cause differences in illumination across images (a \
to b), which may lead to differences in measurement sizes.\n\nThe predicted subfigure \
names are: {"names":[{"name":"a"},{"name":"b"},{"name":"c"}]}.\n',
        "assistant": SplitSubCaptions.model_validate(
            {
                "sub_captions": {
                    "a": {
                        "text": "Rate of preserved autofluorescence (PAF) area loss \
seen in fundus autofluorescence (FAF) images. Measurement of the PAF area at baseline. \
The foveal area remains intact with consistent PAF, surrounded by an area of mottled \
AF. Alignment of the patient to camera while imaging can cause differences in \
illumination across images, which may lead to differences in measurement sizes."
                    },
                    "b": {
                        "text": "Rate of preserved autofluorescence (PAF) area loss \
seen in fundus autofluorescence (FAF) images. The second follow-up timepoint is from \
four years later. Alignment of the patient to camera while imaging can cause \
differences in illumination across images, which may lead to differences in \
measurement sizes."
                    },
                    "c": {
                        "text": "Rate of preserved autofluorescence (PAF) area loss \
seen in fundus autofluorescence (FAF) images. The baseline PAF area is shown as the \
solid line, while follow up PAF is a dashed line."
                    },
                }
            }
        ),
    },
    {  # PMC3991579_pone.0093624.g002
        "user": 'Preprocessing: (from left to right) Original image (A), field of view \
mask (B) and enhanced image after uneven illumination correction (C).\n\nThe predicted \
subfigure names are: {"names":[{"name":"A"},{"name":"B"},{"name":"C"}]}.\n',
        "assistant": SplitSubCaptions.model_validate(
            {
                "sub_captions": {
                    "A": {"text": "Preprocessing: Original image."},
                    "B": {"text": "Preprocessing: field of view mask."},
                    "C": {
                        "text": """Preprocessing: enhanced image after uneven \
illumination correction."""
                    },
                }
            }
        ),
    },
]


def create_schema_message(model: Type[pydantic.BaseModel]) -> str:
    """
    Convert a pydantic model into a string describing the response schema.

    Args:
        model (Type[pydantic.BaseModel]): Pydantic model describing the response schema.

    Returns:
        str: String describing the response schema.

    """
    schema = model.model_json_schema()
    schema_message = f"""
Return the answer using the following json schema: {schema}. Only include the JSON in \
your response, no other text. Do not use markdown formatting.
"""
    return schema_message + "\n"


def get_messages(
    caption: str,
    system_message: str,
    few_shot_examples: Sequence[dict[str, Any]] | None = None,
    response_schema_message: str | None = None,
) -> list[dict[str, str]]:
    """
    Convert the caption and other information into messages for the LLM.

    Args:
        caption (str): Caption to be processed.
        system_message (str): System message for the LLM.
        few_shot_examples (list[dict[str, str]] | None, optional): Few shot examples
            to include. A list of dictionaries with the keys "user" for the user
            message and "assistant" for the assistant response. Defaults to None.
        response_schema_message (str | None, optional): String describing the response
            schema. Defaults to None.

    Returns:
        list[dict[str, str]]: The messages for the LLM.

    """
    if response_schema_message is None:
        response_schema_message = ""

    messages = [
        {
            "role": "system",
            "content": system_message + response_schema_message,
        },
    ]

    if few_shot_examples is not None:
        for example in few_shot_examples:
            messages.append(
                {
                    "role": "user",
                    "content": example["user"]
                    if isinstance(example["user"], str)
                    else example["user"].model_dump_json(),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": example["assistant"]
                    if isinstance(example["assistant"], str)
                    else example["assistant"].model_dump_json(),
                }
            )
    messages.append({"role": "user", "content": caption})
    return messages
