"""Visual diagnostic of exact SAM input and masks before/after overlap processing."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from spectral_anomaly import analyze_spectrum, load_config, postprocess_masks, segment_spectrum

class DemoGenerator:
    def generate(self,image):
        a=np.zeros(image.shape[:2],bool); a[8:25,10:35]=1; b=np.zeros_like(a); b[9:26,11:36]=1
        return [{"segmentation":x,"area":int(x.sum()),"bbox":[10,8,26,18],"predicted_iou":.9,"stability_score":.9} for x in (a,b)]
class DemoAutomatic:
    def generate_masks(self,image):
        from spectral_anomaly.sam_segmentation import annotations_to_segments
        return annotations_to_segments(DemoGenerator().generate(image))

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--spectral-config",default="configs/spectral_analysis.json"); p.add_argument("--sam-config",default="configs/sam.json"); p.add_argument("--output",default="sam_diagnostic.html"); p.add_argument("--real-sam",action="store_true")
    a=p.parse_args(argv); sc=load_config(a.spectral_config,"spectral"); sam=load_config(a.sam_config,"sam")
    t=np.arange(sc["windowing"]["size"])/sc["sampling_frequency"]; spectrum=analyze_spectrum(np.sin(2*np.pi*.08*t),sc)
    image,raw,points=segment_spectrum(spectrum,sam,automatic_segmenter=None if a.real_sam else DemoAutomatic())
    final=postprocess_masks(raw,**sam["mask_postprocessing"])
    fig=make_subplots(rows=1,cols=4,subplot_titles=("Représentation","Entrée SAM","Avant","Après"))
    fig.add_trace(go.Heatmap(z=np.log1p(abs(spectrum.values))),1,1); fig.add_trace(go.Image(z=image),1,2)
    for col,items in ((3,raw),(4,final)):
        overlay=sum((s.mask.astype(int)*s.segment_id for s in items),np.zeros(spectrum.values.shape,int)); fig.add_trace(go.Heatmap(z=overlay),1,col)
    fig.update_layout(title=f"Fusion: {[s.metadata.get('overlap_links',[]) for s in final]}"); fig.write_html(a.output); return raw,final
if __name__ == "__main__": main()
