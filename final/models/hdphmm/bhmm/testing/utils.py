import time
import numpy as np

from final.models.hdphmm.bhmm import bhmm
from final.models.hdphmm.hdphmmwl import hdphmmwl


def filter_data_with_labels(some_data, some_labels, label):
    whale_dat_indicis = np.where(some_labels == label)[0]
    filtered_data = []
    for i in range(len(some_data)):
        if i in whale_dat_indicis:
            filtered_data.append(some_data[i])
    return filtered_data

def train_model(model_type, n_components, train_data, val_data, hmm_kwargs_arr):

    trained_model = {}
    start_time = time.time()

    if model_type == 'bhmm':
        my_bhmm = bhmm.BayesianHMM(X=np.concatenate(train_data), K=n_components,
                                   **dict(sum(map(list, [hmm_kwargs.items() for hmm_kwargs in hmm_kwargs_arr]), [])))
        curr_hmm = my_bhmm.fit()
        # save
        trained_model['model'] = curr_hmm
        trained_model['elbo'] = 0
        trained_model['bnpy_model'] = my_bhmm
        trained_model['bnpy_hist'] = {}
    elif model_type == 'wl_hdphmm':
        my_hdphmm_wl = hdphmmwl.HDPHMMWL(np.concatenate(train_data), K=n_components,
                                         **dict(sum(map(list, [hmm_kwargs.items() for hmm_kwargs in hmm_kwargs_arr]), [])))
        curr_hmm = my_hdphmm_wl.fit_multiple(verbose=True)
        # hmmdiag_trained_model, hmmdiag_info_dict = bnpy.run(
        # train_data_bnpy, 'FiniteHMM', 'DiagGauss', 'EM',
        # output_path='/tmp/mocap6/showcase-K=20-model=HDPHMM+DiagGauss-ECovMat=1*eye/',
        # **dict(
        #     sum(map(list, [hmm_kwargs.items() for hmm_kwargs in hmm_kwargs_arr]), [])))
        # model = get_hmm_learn_from_bnpy(hmmdiag_trained_model)
        # save
        trained_model['model'] = curr_hmm
        trained_model['elbo'] = 0
        trained_model['bnpy_model'] = my_hdphmm_wl
        trained_model['bnpy_hist'] = {}

    # val
    trained_model['val'] = trained_model["model"].score(np.concatenate(val_data))
    trained_model['bic'] = trained_model["model"].bic(np.concatenate(val_data))
    trained_model['aic'] = trained_model["model"].aic(np.concatenate(val_data))

    # N comps
    trained_model["final_comps"] = trained_model["model"].n_components

    # Stats
    trained_model["time"] = time.time() - start_time

    return trained_model

def get_all_results(model_types, num_components_to_test, whale_labels, whale_data, test_args, bnpy_kwargs_arr):

    n_inits = test_args['n_inits']
    cv_amt = test_args['cv_amt']

    all_whale_results = {}

    for whale_label in whale_labels:

        print('testing for whale type: ', whale_label)
        all_results = {}

        for model_ind in range(len(model_types)):

            print('testing for model_type: ', model_types[model_ind])
            model_results = {}
            results_per_component = {}  # results per dimension
            start_outer = time.time()
            cv_len = 0

            for num_comps in num_components_to_test:

                cv_len = len(whale_data['train_data'])
                cv_test = whale_data['test_data']

                # bnpy_kwargs_arr.append(dict(K=num_comps))
                trained_models = []
                total_inits = 0

                for cv_index in range(cv_amt):

                    train_for_whale = filter_data_with_labels(whale_data['train_data'][cv_index],whale_data['train_label'][cv_index], whale_label)
                    val_for_whale = filter_data_with_labels(whale_data['val_data'][cv_index],whale_data['val_label'][cv_index], whale_label)

                    for i in n_inits:
                        print('_______________________________')
                        print('whale: ', whale_label,
                              ' mode: ', model_types[model_ind],
                              ' num comps: ', num_comps,
                              ' cv_index: ', cv_index,
                              ' n_inits: ', i,'/',n_inits
                              )
                        print('_______________________________')
                        total_inits += 1
                        model_it = None
                        try:
                            model_it = train_model(
                                model_types[model_ind],
                                num_comps,
                                train_for_whale,
                                val_for_whale,
                                bnpy_kwargs_arr,
                            )

                        # Code that may raise an exception
                        # ...
                        except ValueError as e:
                            print(e)
                        # Code to handle the exception
                        else:
                            trained_models.append(model_it)
                        # Code to be executed if no exception occurs in the try block
                        finally:
                            pass
                            # Code that will be executed no matter what, whether an exception occurs or not
                            #    ...

                best_model_ind = np.argmax([trained_model['val'] for trained_model in trained_models])
                best_model = trained_models[best_model_ind]['model']
                average_score = np.mean([trained_model['val'] for trained_model in trained_models])

                results_per_component[num_comps] = {
                    'lls': [trained_model['val'] for trained_model in trained_models],
                    'elbos': [trained_model['elbo'] for trained_model in trained_models],
                    'models': [trained_model['model'] for trained_model in trained_models],
                    'bnpy_model': [trained_model['bnpy_model'] for trained_model in trained_models],
                    'bnpy_hist': [trained_model['bnpy_hist'] for trained_model in trained_models],
                    'test': best_model.score(np.concatenate(cv_test)),
                    'avg_val': average_score,
                    'final_components': best_model.n_components,
                    'final_components_avg': np.mean([trained_model['model'].n_components for trained_model in trained_models]),
                    'its': total_inits,
                    'time': [trained_model['time'] for trained_model in trained_models],
                    'aic': np.mean([trained_model['aic'] for trained_model in trained_models]),
                    'bic': np.mean([trained_model['bic'] for trained_model in trained_models]),
                    'best_model': best_model
                }

                print('completed, final component avg: ', results_per_component[num_comps]['final_components_avg'])

            end_outer = time.time()

            model_results['total_time'] = end_outer - start_outer
            model_results['components'] = results_per_component
            model_results['total_its'] = cv_len * len(n_inits)
            model_results['component_list'] = num_components_to_test

            all_results[model_types[model_ind]] = model_results

        all_whale_results[whale_label] = all_results

    return all_whale_results
