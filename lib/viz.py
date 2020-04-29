#
# Visualization tools
#

import io
import weblogo
from IPython.display import Image

def sequence_compare(a, b):
    print(a)
    print(''.join(['|' if a[i] != b[i] else ' ' for i in range(len(a))]))
    print(b)

def render_weblogo(seqs, residues_per_line=100):
    fin = io.StringIO()
    for i in range(len(seqs)):
        fin.write(f"> seq_{i}\n")
        fin.write(seqs[i] + "\n")
    fin.seek(0)

    seqs = weblogo.read_seq_data(fin)
    logodata = weblogo.LogoData.from_seqs(seqs)
    logooptions = weblogo.LogoOptions()
    logooptions.unit_name = 'probability'
    logooptions.stacks_per_line = residues_per_line
    logooptions.color_scheme = weblogo.colorscheme.chemistry
    logoformat = weblogo.LogoFormat(logodata, logooptions)
    return Image(weblogo.png_print_formatter(logodata, logoformat))