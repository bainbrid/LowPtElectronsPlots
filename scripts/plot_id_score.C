{

  TFile* f = new TFile("../data/170823/nonres_large/output_0.root");
  //TFile* f = new TFile("../data/output.LATEST.root");
  
  TTree* t = (TTree*)f->Get("ntuplizer/tree");
  TCanvas* c = new TCanvas();
  
  TString cuts = "1";
  //cuts += " && tag_pt > 7.0 && abs(tag_eta) < 1.5"; // Tag-side muon
  cuts += " && trk_pt > 0.5 && abs(trk_eta) < 2.5"; // reco'ed as an ele and in acc
  
  // Algo type and ID version
  int version = 3;
  TString var = "";
  TString suffix = "";
  if        (version == 0) { // PF default
    var = "ele_mva_value";
    cuts += " && ele_mva_value > -99. && is_egamma==1";
    suffix = "_pf_old";
  } else if (version == 1) { // PF retrained
    var = "ele_mva_value_retrained";
    cuts += " && ele_mva_value_retrained > -99. && is_egamma==1";
    suffix = "_pf_new";
  } else if (version == 2) { // LP old
    var = "ele_mva_value";
    cuts += " && has_gsf && ele_mva_value > -99. && is_egamma==0";
    suffix = "_lp_old";
  } else if (version == 3) { // LP new
    var = "ele_mva_value_depth10";
    cuts += " && has_gsf && ele_mva_value_depth10 > -99. && is_egamma==0";
    suffix = "_lp_new";
  }
  //std::cout << "cuts: " << cuts << std::endl;
				   
  TH1F s("s","",50,-10.,10.);
  TString cuts_s = cuts + TString(" && is_e==1");
  t->Draw(var+">>s",cuts_s,"hist");
  s.SetTitle(cuts);
  s.SetLineColor(kGreen+2);
  s.SetLineWidth(2);
  s.Scale(1.0 / s.Integral());
  
  TH1F b("b","",50,-10.,10.);
  TString cuts_b = cuts + TString(" && is_e==0");
  t->Draw(var+">>b",cuts_b,"hist");
  b.SetLineColor(kRed);
  b.SetLineWidth(2);
  b.Scale(1.0 / b.Integral());

  Double_t max = s.GetMaximum() > b.GetMaximum() ? s.GetMaximum() :b.GetMaximum();
  s.SetMaximum(max*1.1);
  s.GetXaxis()->SetTitle(suffix);

  //s.SetMinimum(0.5);
  //c->SetLogy();
    
  s.Draw("hist");
  b.Draw("hist same");

  gStyle->SetOptStat(0);
  c->SaveAs("plot"+suffix+".pdf");

}

