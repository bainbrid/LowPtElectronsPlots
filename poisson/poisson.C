#include "TH1.h"
#include <iostream>
{
  TH1D * h1 = new TH1D("h1","h1",20,0.,20.);
   h1->Sumw2(kFALSE);
   h1->SetBinErrorOption(TH1::kPoisson);
   for ( int ibin = 0; ibin < 20; ++ibin ) { h1->SetBinContent(ibin+1,ibin*1.); }

    //example: lower /upper error for bin 20
   for ( int ibin = 0; ibin < 20; ++ibin ) {
     //int ibin = 20;
     double err_low = h1->GetBinErrorLow(ibin);
     double err_up = h1->GetBinErrorUp(ibin);
     std::cout << ibin << " " << err_low << " " << err_up << std::endl;
   }

   // draw histogram with errors 
   // use the drawing option "E0" for drawing the errors
   // for the bin with zero content

   h1->SetMarkerStyle(20);
   h1->Draw("E");
}

//0 0 1.84102
//1 0 1.84102
//2 0.827246 2.29953
//3 1.29181 2.63786
//4 1.6327 2.91819
//5 1.91434 3.16275
//6 2.15969 3.38247
//7 2.37993 3.58364
//8 2.58147 3.77028
//9 2.76839 3.94514
//10 2.94346 4.1102
//11 3.10869 4.26695
//12 3.26558 4.41652
//13 3.41527 4.55982
//14 3.55866 4.69757
//15 3.6965 4.83038
//16 3.82938 4.95874
//17 3.9578 5.08307
//18 4.08218 5.20372
//19 4.20289 5.32101
