import ROOT as r
import numpy as np

################################################################################

def draw_root_wp(eff,fr,**kwargs):
    gr = r.TGraph()
    r.SetOwnership(gr,False)
    gr.SetPoint(0,fr,eff)
    gr.SetMarkerStyle(kwargs.get('style',20))
    gr.SetMarkerSize(kwargs.get('size',2))
    gr.SetMarkerColor(kwargs.get('color',r.kBlack))
    gr.SetName(kwargs.get('label','unknown'))
    gr.SetTitle('')
    gr.Draw("Psame")

################################################################################

def draw_root_roc(tpr,fpr,auc,eff=1.,fr=1.,**kwargs):
    #print("len(fpr)",len(fpr))
    step = max(1,int(len(fpr)/1000.))
    fpr = fpr[::step]
    tpr = tpr[::step]
    gr = r.TGraph(len(fpr), fpr*fr, tpr*eff)
    r.SetOwnership(gr,False)
    gr.SetLineStyle(kwargs.get('style',1))
    gr.SetLineWidth(kwargs.get('width',2))
    gr.SetLineColor(kwargs.get('color',r.kBlack))
    label = kwargs.get('label','unknown')
    if auc is not None: label += ' ({:.3f})'.format(auc)
    gr.SetName(label)
    gr.SetTitle('')
    gr.Draw("Lsame")

################################################################################

def draw_root_score(data,**kwargs):

    # Create histo
    title = kwargs.get('title','draw_root_score')
    nbin = kwargs.get('nbin',100)
    xmin = kwargs.get('xmin',np.min(data))
    xmax = kwargs.get('xmax',np.max(data))
    his = r.TH1F(title,'',nbin,xmin,xmax)
    r.SetOwnership(his,False)

    # Fill histo
    for val in data:
        val = min(max(val,xmin+1.e-9),xmax-1.e-9)
        his.Fill(val)

    # Axes
    width = ( xmax - xmin ) / nbin
    dp = max(0,int(np.log10(1./width)))
    xtitle = kwargs.get('xtitle','Unknown')
    xunit = kwargs.get('xunit',None)
    ytitle = kwargs.get('ytitle',f'Entries / {width}')
    if xunit is not None and xunit is not "": xtitle += f" [{xunit}]"
    if xunit is not None and xunit is not "": ytitle += f" {xunit}"
    his.GetXaxis().SetTitle(xtitle)
    his.GetYaxis().SetTitle(ytitle)
    his.GetYaxis().SetMaxDigits(3)
    his.GetYaxis().SetNdivisions(505)

    # Ranges
    norm = kwargs.get('norm',True)
    logy = kwargs.get('logy',False)
    ymin = 0.
    ymin = 1.
    if norm:
        scale = 1./his.Integral()
        his.Scale(scale)
        ymin = 10**int(np.log10(scale))
        ymin = kwargs.get('ymin',ymin if logy else 0.)
    else:
        ymin = kwargs.get('ymin',0.1 if logy else 0.)
    ymax = kwargs.get('ymax',his.GetMaximum()*2. if logy else his.GetMaximum()*1.2)
    his.SetMinimum(ymin)
    his.SetMaximum(ymax)

    # Drawing
    data = kwargs.get('is_data',False)
    if data == False:
        his.SetLineStyle(kwargs.get('style',1))
        his.SetLineWidth(kwargs.get('width',2))
        his.SetLineColor(kwargs.get('color',r.kBlue))
    else:
        his.SetMarkerStyle(20)
        his.SetMarkerSize(1)
        his.SetMarkerColor(r.kBlack)
        his.SetLineStyle(1)
        his.SetLineWidth(2)
        his.SetLineColor(r.kBlack)
        
    same = kwargs.get('same',False)
    if data == False:
        his.Draw("HIST same" if same else "HIST")
    else:
        his.Draw("P E X0 same" if same else "P E X0")

################################################################################

def cmsLabels(
        pad,
        lumiText,
        extraText=None,
        xPos=0.16,
        yPos=0.83,
        xPosOffset=0.1,
        yPosOffset=0.,
        ):
    
    # Latex box
    latex = r.TLatex()
    latex.SetNDC()
    latex.SetTextAngle(0)
    latex.SetTextColor(r.kBlack)
    pad.cd()
    top = pad.GetTopMargin()
    right = pad.GetRightMargin()
    
    # CMS (Preliminary) label
    cmsText = "CMS"
    cmsTextFont = 61
    cmsTextSize = 0.75
    latex.SetTextFont(cmsTextFont)
    latex.SetTextSize(cmsTextSize*top)
    latex.SetTextAlign(11)
    latex.DrawLatex(xPos, yPos, cmsText)

    if extraText != None and extraText != "":
        #extraText = "Preliminary"
        extraTextFont = 52 
        extraTextSize = 0.75*cmsTextSize
        latex.SetTextFont(extraTextFont)
        latex.SetTextSize(extraTextSize*top)
        latex.SetTextAlign(11)
        latex.DrawLatex(xPos+xPosOffset, yPos+yPosOffset, extraText)

    # CMS lumi label
    #lumiText = "33.9 fb^{-1} (13 TeV)"
    #lumiText = "scale[0.85]{"+lumiText+"}"
    lumiTextSize = 0.6
    lumiTextOffset = 0.2
    latex.SetTextFont(42)
    latex.SetTextAlign(31)
    latex.SetTextSize(lumiTextSize*top)
    latex.DrawLatex(1-right,1-top+lumiTextOffset*top,lumiText)
    #pad.Update()
