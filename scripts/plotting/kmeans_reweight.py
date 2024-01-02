import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt
import json

def calc_weights(
        data,
        reweight_features = ['log_trkpt', 'trk_eta'],
        nbins=600,
        dataset="test",
        base = "/Users/bainbrid/Repositories/LowPtElectronsPlots/scripts",
        tag='2023Dec15',
        filename='test_kmeans_weights') :

    # Vectorized method
    apply_weight = np.vectorize(lambda x,y : y.get(x),excluded={1})

    models = f'{base}/models/{tag}'
    if not os.path.isfile(f'{models}/{dataset}_kmeans_weights.json') or not os.path.isfile(f'{models}/{dataset}_kmeans_weights.pkl') :
        print(f'Unable to find "{models}/{dataset}_kmeans_weights.json" and "{models}/{dataset}_kmeans_weights.pkl"')
        return kmeans_reweight(
            data,
            reweight_features=reweight_features,
            from_file=False,
            tag=tag,
            )

    if 'log_trkpt' in reweight_features: data['log_trkpt'] = np.log10(data.trk_pt)
    import joblib
    print(f'Retrieving weights and binning from "{models}/{dataset}_kmeans_weights.json" and "{models}/{dataset}_kmeans_weights.pkl"...')
    model = joblib.load(f'{models}/{dataset}_kmeans_weights.pkl')
    bin = model.predict(data[reweight_features])
    weights_json = json.load(open(f'{models}/{dataset}_kmeans_weights.json'))
    weights_lut = {}
    for i in weights_json :
        try : weights_lut[int(i)] = weights_json[i]
        except : success = False
    weights = apply_weight(bin,weights_lut)
    label='is_e' # ["is_e","is_data"][0 if only_mc else 1] #@@ ?????????????
    data['weight'] = weights * np.invert(data[label]) + data[label]
    return data,model
    
def train_test_split(data, div, thr):
   mask = data.evt % div
   mask = mask < thr
   return data[mask], data[np.invert(mask)]

def kmeans_reweight(
        data,
        reweight_features = ['log_trkpt', 'trk_eta'],
        nbins=600,
        dataset="test",
        base = "/Users/bainbrid/Repositories/LowPtElectronsPlots/scripts",
        tag="2023Dec15",
        from_file=True,
        ):

    #vectorize(excluded={2})
    apply_weight = np.vectorize(lambda x, y: y.get(x), excluded={2})
    
    models = f'{base}/models/{tag}'
    if not os.path.isdir(models):
        os.makedirs(models)

    plots = f'{base}/models/{tag}'
    if not os.path.isdir(plots):
        os.makedirs(plots)

    # WHAT ARE WE REWEIGHTING? BKGD TO SIGNAL? MC TO DATA?
    data = data.astype({'is_mc':'bool'})
    data["is_data"] = np.invert(data.is_mc)
    data = data.astype({'is_data':'bool'})
    only_mc = np.all(data['is_mc'])
    label = ["is_e","is_data"][0 if only_mc else 1]
        
    if 'log_trkpt' in reweight_features: data['log_trkpt'] = np.log10(data.trk_pt)
    data['original_weight'] = 1. #np.invert(label)*original_weight.get_weight(data.log_trkpt, data.trk_eta)+label

    overall_scale = data.shape[0]/float(data[label].sum())

    if from_file:

        data,clusterizer = calc_weights(
            data=data,
            reweight_features=reweight_features,
            nbins=nbins,
            label=label,
            base=base,
            tag=tag,
            filename='test_kmeans_weights',
        )

    else:

        print('Clustering ....................')
        from sklearn.cluster import KMeans, MiniBatchKMeans
        clusterizer = MiniBatchKMeans(
            n_clusters=nbins,
            #init='random',
            max_no_improvement=None,
            #batches=3000,
            #n_jobs=3
            verbose=False,
            ) 
        clusterizer.fit(data[reweight_features]) #fit(data[label][reweight_features])
        print('Done!')

        global_ratio = float(data[label].sum())/np.invert(data[label]).sum() # what is this?

        data['cluster'] = clusterizer.predict(data[reweight_features])
        counts = {}
        weights = {}
        for cluster, group in data.groupby('cluster'):
            nbkg = np.invert(group[label]).sum()
            nsig = group[label].sum()
            if not nbkg: RuntimeError('cluster {nbkg} has no background events, reduce the number of bins!')
            elif not nsig: RuntimeError('cluster {nsig} has no electrons events, reduce the number of bins!')
            weight = float(nsig)/nbkg if nbkg > 0 else 1.
            weights[cluster] = weight
            counts[cluster] = min(nsig,nbkg)
        print("Number of signal: ",data[label].sum())
        print("Number of bkgd:",np.invert(data[label]).sum())
    
        import joblib
        joblib.dump(
            clusterizer, 
            f'{models}/{dataset}_kmeans_weights.pkl',
            compress=True
            )
    
        weights['features'] = reweight_features
        print(f'{models}/{dataset}_kmeans_weights.json')
    
        with open(f'{models}/{dataset}_kmeans_weights.json', 'w') as ww:
            json.dump(weights, ww)
        print('...done')
        del weights['features']
    
        data['weight'] = np.invert(data[label])*apply_weight(data.cluster, weights)+data[label]
        
        print('time for plots!')
        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = 0.01 # point in the mesh [x_min, x_max]x[y_min, y_max].

        xfeature = reweight_features[0]
        yfeature = reweight_features[1]
        
        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = data[xfeature].min() - 0.3, data[xfeature].max() + 0.3
        y_min, y_max = data[yfeature].min() - 0.3, data[yfeature].max() + 0.3
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h, dtype=np.float32),
            np.arange(y_min, y_max, h, dtype=np.float32),
            )
    
        # Obtain labels for each point in mesh. Use last trained model.
        Zlin = clusterizer.predict(np.c_[xx.ravel(), yy.ravel()])
    
        # Put the result into a color plot
        #import cosmetics
        Z = Zlin.reshape(xx.shape)
        plt.figure(figsize=[8, 8])
        plt.clf()
        plt.imshow(
            Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect='auto', origin='lower')
        plt.title('weighting by clustering')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel(xfeature)#cosmetics.beauty['log_trkpt'])
        plt.ylabel(yfeature)#cosmetics.beauty['trk_eta'])
        plt.plot()
        #try : plt.savefig(f'{plots}/{dataset}_clusters.png')
        #except Exception as e: print(e)
        print("here",f'{plots}/{dataset}_clusters.pdf')
        try : plt.savefig(f'{plots}/{dataset}_clusters.pdf')
        except Exception as e: print(e)
        plt.clf()
        
        from matplotlib.colors import LogNorm
        Z = apply_weight(Zlin, weights).reshape(xx.shape)
        plt.figure(figsize=[10, 8])
        plt.imshow(
            Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.seismic,
            norm=LogNorm(vmin=10**-4, vmax=10**4),
            aspect='auto', origin='lower')
        plt.title('weight')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel(xfeature)#cosmetics.beauty['log_trkpt'])
        plt.ylabel(yfeature)#cosmetics.beauty['trk_eta'])
        plt.colorbar()
        plt.plot()
        #try : plt.savefig(f'{plots}/{dataset}_clusters_weights.png')
        #except Exception as e: print(e)
        try : plt.savefig(f'{plots}/{dataset}_clusters_weights.pdf')
        except Exception as e: print(e)
        plt.clf()
    
        from matplotlib.colors import LogNorm
        Z = apply_weight(Zlin, counts).reshape(xx.shape)
        plt.figure(figsize=[10, 8])
        plt.imshow(
            Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.seismic,
            norm=LogNorm(vmin=0.1, vmax=max(counts.values())*10.),
            aspect='auto', origin='lower')
        plt.title('counts')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel(xfeature)#cosmetics.beauty[])
        plt.ylabel(yfeature)#cosmetics.beauty[])
        plt.colorbar()
        plt.plot()
        #try : plt.savefig(f'{plots}/{dataset}_clusters_counts.png')
        #except Exception as e: print(e)
        try : plt.savefig(f'{plots}/{dataset}_clusters_counts.pdf')
        except Exception as e: print(e)
        plt.clf()
    
        # plot weight distribution
        weights = data.weight[np.invert(data[label])]
        entries, _, _ = plt.hist(
            weights, 
            bins=np.logspace(
                np.log(max(weights.min(), 10**-5)),
                np.log(weights.max()*2.),
                100
                ),
            histtype='stepfilled'
        )
    
        plt.xlabel('Weight')
        plt.ylabel('Occurrency')
        plt.legend(loc='best')
        plt.ylim(0.5, entries.max()*10.)
        plt.xlim(max(weights[weights>0.].min()*0.5,10**-3), weights.max()*2.)
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        plt.plot()
        #try : plt.savefig(f'{plots}/{dataset}_clustering_weights.png')
        #except Exception as e: print(e)
        sum_of_weights = weights.sum()
        sum_of_events = np.invert(data[label]).sum()
        print("test",f"sum of weights: {sum_of_weights}")
        print("test",f"number of entries: {sum_of_events}")
        print("test",f"average weight: {sum_of_weights/sum_of_events}")
        plt.text(
            plt.xlim()[0]+0.1*(plt.xlim()[1]-plt.xlim()[0]),
            plt.ylim()[0]+0.2*(plt.ylim()[1]-plt.ylim()[0]),
            f"sum of weights: {sum_of_weights:.1f}\n"\
            f"number of entries: {sum_of_events:.0f}\n"\
            f"average weight: {sum_of_weights/sum_of_events:.3f}"
        )
        try : plt.savefig(f'{plots}/{dataset}_clustering_weights.pdf')
        except Exception as e: print(e)
        plt.clf()
        
        for var in reweight_features: #+['trk_pt']:
            #x_range =
            #min(data[label][var].min(), data[np.invert(label)][var].min()), 
            #max(data[label][var].max(), data[np.invert(label)][var].max())
            #x_range = cosmetics.ranges.get(var, x_range)
            x_range = (data[var].min()-0.3,data[var].max()+0.3)
            #x_range = {
            #"trk_pt":(0.,15.),
            #"trk_eta":(-3.,3.),
            #"log_trkpt":(0.,1.3),
            #}.get(var)
            for name, weight, ls in [
                #('unweighted', np.ones(data.shape[0]), "dotted"),
                ('unweighted', data.original_weight, "solid"),
                ('weighted', data.weight, "dashed"),
            ]:
                tmp = {'is_e':'signal (MC), ','is_data':'bkgd (data), '}.get(label) # values = True
                color = {'is_e':'green','is_data':'red'}.get(label)
                plt.hist(
                    data[data[label]][var], bins=100, density=True, linewidth=1.2, edgecolor=color, linestyle=ls,
                    histtype='step', label=tmp+name, range=x_range, weights=weight[data[label]]
                )
                tmp = {'is_e':'bkgd (MC), ','is_data':'signal (MC), '}.get(label) # values = False
                color = {'is_e':'red','is_data':'green'}.get(label)
                plt.hist(
                    data[np.invert(data[label])][var], bins=100, density=True, linewidth=1.2, edgecolor=color, linestyle=ls,
                    histtype='step', label=tmp+name, range=x_range, weights=weight[np.invert(data[label])]
                )
            plt.legend(loc='best')
            plt.xlabel(var)# if var not in cosmetics.beauty else cosmetics.beauty[var])
            plt.ylabel('A.U.')   
            #try : plt.savefig(f'{plots}/{dataset}_{var}.png')
            #except Exception as e: print(e)
            try : plt.savefig(f'{plots}/{dataset}_{var}.pdf')
            except Exception as e: print(e)
            plt.clf()
            
        #compute separation with a BDT   
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import roc_curve, roc_auc_score
    
        train_bdt, test_bdt = train_test_split(data, 10, 5)
    
        print('Classifying without weights ....................')
        pre_separation = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            verbose=0,
        )
        pre_separation.fit(train_bdt[reweight_features], train_bdt[label])
        print('Done!')

        test_proba = pre_separation.predict_proba(test_bdt[reweight_features])[:, 1]
        roc_pre = roc_curve(test_bdt[[label]],  test_proba)[:2]
        auc_pre = roc_auc_score(test_bdt[[label]],  test_proba)
    
        print('Classifying with weights ....................')
        post_separation = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            verbose=0,
        )
        post_separation.fit(train_bdt[reweight_features], train_bdt[label], train_bdt.weight)
        print('Done!')

        test_proba = post_separation.predict_proba(test_bdt[reweight_features])[:, 1]
        roc_post = roc_curve(test_bdt[[label]],  test_proba, sample_weight=test_bdt.weight)[:2]
        auc_post = roc_auc_score(test_bdt[[label]],  test_proba, sample_weight=test_bdt.weight)
    
        # make plots
        plt.clf()
        plt.figure(figsize=[8, 8])
        plt.plot(*roc_pre, label=f'Unweighted (AUC={auc_pre:.3f})')
        plt.plot(*roc_post, label=f'Weighted (AUC={auc_post:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Mistag Rate')
        plt.ylabel('Efficiency')
        plt.legend(loc='best')
        plt.plot()
        #plt.savefig(f'{plots}/{dataset}_reweighting.png')
        plt.savefig(f'{plots}/{dataset}_reweighting.pdf')
        plt.clf()
        
        return data,clusterizer

