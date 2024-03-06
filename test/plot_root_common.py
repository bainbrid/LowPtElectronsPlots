import ROOT as r

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
    print("len(fpr)",len(fpr))
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
