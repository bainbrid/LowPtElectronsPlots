import uproot 

t = uproot.open("~/Desktop/MINIAODSIM.root")["Events"]

for basket in (t["floatedmValueMap_lowPtGsfElectronID__PAT.obj.values_"].iterate_baskets()): # ids_
    print(basket)

#for basket in (t["floatedmValueMap_lowPtGsfElectronID__RECO.obj.values_"].iterate_baskets()): # ids_
#    print(basket)

#for basket in (t["patElectrons_slimmedLowPtElectrons__PAT.obj"].iterate_baskets()): # ids_
#    print(basket)
