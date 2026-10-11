# Licenses and notices

Tennis3D code and the lifter weights trained by this project are licensed
GPL-3.0-only. The full text is in the root [LICENSE](../LICENSE) file.

Third-party material keeps its own terms. These notices must stay with any
copy of the code:

| Material | Terms | Notice |
| --- | --- | --- |
| Simulation and dense lifting code adapted from [KieDani/UpliftingTableTennis](https://github.com/KieDani/UpliftingTableTennis), revision 1a64bf6488f3a142f733e73e8de00bfecf044b23 | GPL-3.0 | Uplifting-GPL-3.0.txt, unchanged upstream license |
| Google Sans Code typeface, Latin subset from googlefonts/googlesans-code v7.001, embedded in the app page, offline viewer and labeling dashboard | SIL Open Font License 1.1 | Google-Sans-Code-OFL-1.1.txt and tennis3d/resources/theme/OFL.txt |
| Three.js 0.180.0, jsonc-parser 3.3.1, Ajv 8.20.0 and fast-uri, bundled or used to build the viewer page | MIT and BSD-3-Clause, as each package states | Full texts are embedded in the built HTML under "About and licenses" |

The adapted modules name their upstream source in their headers. The lifter
adapts the RoPE, residual blocks and XYZ head of upstream uplifting/model.py,
and its ratio-four feed-forward layer differs from the released ratio-one
implementation.

The font stays under the OFL. "Google Sans Code" is a trademark of Google LLC.
Its use here names the typeface and implies no affiliation.

Match footage, including the example clips in examples/gallery-v2, detector
weights fetched from their authors' links and Python dependencies installed
from their own distributions keep their owners' terms.
