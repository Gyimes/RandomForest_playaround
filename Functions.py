# Functions for ML building
# Packages
import pandas as pd
import numpy as np
import difflib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, accuracy_score, recall_score, roc_auc_score
import xgboost as xgb
# Organise the data
def loader():
    while True:
        file_path = input("Enter the path to the data: ")
        try:
            raw = pd.ExcelFile(file_path)
            break
        except FileNotFoundError:
            print("Error: The file was not found.")
        except Exception as e:
            print(f"Error: {e}")
    sn = raw.sheet_names
    for i, sheet in enumerate(sn):
        print(f"{i + 1}. {sheet}")
        
    while True:
        try:
            si = input(f"Enter the sheet numbers you want to load in (separated by commas): ")
            si = [int(i) - 1 for i in si.split(',')]

            if not all(0 <= i < len(sn) for i in si):
                raise ValueError("One or more entered sheet numbers are out of range.")

            sheets = [sn[i] for i in si]
            dfs = {sheet: pd.read_excel(file_path, sheet_name=sheet) for sheet in sheets}
            break
        except ValueError as ve:
            print(f"Error: {ve} Please enter valid sheet numbers.")
        except Exception as e:
            print(f"Error: {e}")
    return dfs, sheets

# Merge the sheets
def merger(dfs, sheets):
    ccolumns = set(dfs[sheets[0]].columns)
    for sheet in sheets[1:]:
        ccolumns.intersection_update(dfs[sheet].columns)
    if ccolumns:
        print(f"Potential ID(s) to merge: {', '.join(ccolumns)}")
        while True:
            merge_option = input("Merge the sheets based on any of these? (y/n): ")
            if merge_option.lower() in['y', 'n', 'yes', 'no']:
                break
            else:
                print(f"Please type y or n")
        if merge_option.lower() == 'y' or merge_option.lower() == 'yes':
            if len(ccolumns) > 1:
                colmerge = input(f"Enter the column to merge on (options: {', '.join(ccolumns)}): ")
            else:
                colmerge = ccolumns
            if colmerge in ccolumns or colmerge == ccolumns:
                merged_df = dfs[sheets[0]]
                original_size = pd.DataFrame({'Sheet name': [sheets[0]], 'Length': [len(dfs[sheets[0]])]})
                for sheet in sheets[1:]:
                    merged_df = pd.merge(merged_df, dfs[sheet], on=next(iter(ccolumns)))
                    original_size = pd.concat([original_size, pd.DataFrame({'Sheet name': [sheet], 'Length': [len(dfs[sheet])]})])
                original_size = pd.concat([original_size, pd.DataFrame({'Sheet name': ['Merged'], 'Length': [len(merged_df)]})])
                print(original_size)
                return merged_df
            else:
                print("Error, no valif column selected to merge")
        elif merge_option.lower() == 'n' or merge_option.lower() == 'no':
            for s in sheets:
                col = input(f"Enter the column to be used to merge (options: {dfs[s].columns}): ")
                dfs[sheet].rename(columns={col: 'merger_col'}, inplace=True)
            for sheet in sheets[1:]:
                merged_df = pd.merge(merged_df, dfs[sheet], on='merger_col')
            return merged_df
    else:
        print("No common columns found in the selected sheets.")

# Filter the data
def filt_crit(df):
    original_length = len(df)
    def find_closest_match(inp, opts):
        closest_match = difflib.get_close_matches(inp, opts, n=1, cutoff=0.1)
        if closest_match:
            return closest_match[0]
        else:
            return None
    
    def contfilt(df, col):
        def limitdefiner():
            while True:
                lims = input("Please define the limits, e.g. [18, 25] or ]18, 25[ (write min/max if there is no min or max limit): ")
                lims = lims.replace(',', ' ')
                parts = lims.split()
                if len(parts) != 2:
                    print("Incorrect number of limits given")
                else:
                    minval = parts[0]
                    maxval = parts[1]

                    minl = 0 if ']' in minval else 1
                    maxl = 0 if '[' in maxval else 1

                    minval = (minval.replace(']', '').replace('[', ''))
                    maxval = (maxval.replace(']', '').replace('[', ''))

                    return minval, maxval, minl, maxl
        def typechecker(val, df, col):
            if df[col].dtype == 'int64':
                val = int(val)
            elif df[col].dtype == 'float64':
                val = float(val)
            else:
                val = val
            return val
        
        minval, maxval, minl, maxl = limitdefiner()
        cols = set(df.columns)
        
        while minval.lower() == 'min' and maxval.lower() == 'max':
            print("No limits defined")
            doublecheck = input("Do you want to:\n 1 - Redefine the limits\n 2 - Choose another column\n 3 - Do not filter\n")
            if doublecheck not in ['1', '2', '3']:
                opts = ['1 - Redefine the limits', '2 - Choose another column', '3 - Do not filter']
                doublecheck = find_closest_match(doublecheck, opts)
                if doublecheck:
                    doublecheck = doublecheck[0]
            
            if doublecheck == '1':
                minval, maxval, minl, maxl = limitdefiner()
            elif doublecheck == '2':
                col = input(f"Please select the column (options: {', '.join(cols)}): ")
            elif doublecheck == '3':
                return df
        
        if minval.lower() == 'min':
            maxval = typechecker(maxval, df, col)
            if maxl == 1:
                df_filt = df[df[col] <= maxval]
            else:
                df_filt = df[df[col] < maxval]
        elif maxval.lower() == 'max':
            minval = typechecker(minval, df, col)
            if minl == 1:
                df_filt = df[df[col] >= minval]
            else:
                df_filt = df[df[col] > minval]
        else:
            maxval = typechecker(maxval, df, col)
            minval = typechecker(minval, df, col)
            if minl == 1:
                if maxl == 1:
                    df_filt = df[(df[col] >= minval) & (df[col] <= maxval)]
                else:
                    df_filt = df[(df[col] >= minval) & (df[col] < maxval)]
            else:
                if maxl == 1:
                    df_filt = df[(df[col] > minval) & (df[col] <= maxval)]
                else:
                    df_filt = df[(df[col] > minval) & (df[col] < maxval)]
        
        return df_filt
    
    def catfilt(df, col):
        lvls = df[col].unique()
        
        print(f"Here is the list of categories of the selected column:\n {lvls}")
        while True:
            intent = input("Would you like to keep or exclude levels? [k/e] ")

            if intent.lower() == 'k':
                tokeep = input(f"Please list the level(s) you want to keep: ")
                try:
                    tokeep = tokeep.replace(',' ' ').split()
                    df_filt = df[df[col].isin(tokeep)]
                    return df_filt
                except TypeError:
                    tokeep = tokeep
                    df_filt = df[df[col] == tokeep]
                    return df_filt
                

            elif intent.lower() == 'e':
                toexclude = input(f"Please list the level(s) you want to exclude: ")
                try:
                    toexclude = toexclude.replace(',' ' ').split()
                    df_filt = df[~df[col].isin(toexclude)]
                    return df_filt
                except TypeError:
                    toexclude = toexclude
                    df_filt = df[df[col] != toexclude]
                    return df_filt
                
            else:
                print("Incorrect option selected")
    
    def na_filter(df, col):
        while True:
            intent = input("Would you like to remove the rows with NaNs or change the NaNs to a certain value? [r/c]")
            if intent.lower() == 'r':
                return df.dropna(subset=[col])
            elif intent.lower() == 'c':
                replacement = input("Please define the replacement value: ")
                return df.fillna(value=replacement)
            else:
                print("Incorrect option selected.")
    
    cols = [item for item in df.columns]
    qe = input(f"Would you like to filter the data? [y/n]")
    if qe.lower() == 'n' or qe.lower() == 'no':
        return df, cols
    col = input(f"Please select the column(s) for filtering (options: {', '.join(cols)}): ")
    
    try:
        col = col.replace(',', ' ').split()
        multicol = True
    except TypeError:
        col = col
        multicol = False
    
    if not multicol:
        critcheck = input("Would you like to filter:\n 1 - Continuous variable\n 2 - Categorical variable\n 3 - NaNs\n")
        if critcheck not in ['1', '2', '3']:
            opts = ['1 - Continuous variable', '2 - Categorical variable', '3 - NaNs']
            critcheck = find_closest_match(critcheck, opts)
            if critcheck:
                critcheck = critcheck[0]    
        filtvers = {'1' : contfilt,
                  '2' : catfilt,
                  '3' : na_filter}
        if critcheck in filtvers:
            df_filt = filtvers[critcheck](df, col)
            n_excluded = original_length - len(df_filt)
            print(f"{n_excluded} cases excluded")
            return df_filt
        else:
            print(f"Weird error")
    if multicol:
        df_filt = df
        for col1 in col:
            critcheck = input(f"Column - {col1}\n Would you like to filter:\n 1 - Continuous variable\n 2 - Categorical variable\n 3 - NaNs\n")
            if critcheck not in ['1', '2', '3']:
                opts = ['1 - Continuous variable', '2 - Categorical variable', '3 - NaNs']
                critcheck = find_closest_match(critcheck, opts)
                if critcheck:
                    critcheck = critcheck[0]    
            filtvers = {'1' : contfilt,
                      '2' : catfilt,
                      '3' : na_filter}
            if critcheck in filtvers:
                df_filt = filtvers[critcheck](df_filt, col1)
                n_excluded = original_length - len(df_filt)
                print(f"{n_excluded} cases excluded")
    return df_filt, cols
      
# Add the functions above into one function
def dataprep():
    # Load in the data
    dfs, sheets = loader()
    
    # merge sheets if needed
    df = merger(dfs, sheets)
    
    # filter data if needed
    df_filt, cols = filt_crit(df)
    
    # Anything to be dummy coded?
    todummy = input('Which, if any of the variables are to be dummy coded?\n ')
    if todummy not in ["", 'no', 'n', 'none']:
        todummy = [item for item in todummy.replace(',',' ').split()]
        df_filt = pd.get_dummies(df_filt, columns=todummy)
    
    # How to code NANs?
    nafill = input('What value, if any should we replace:\n')
    if nafill not in ["", 'no', 'n', 'none']:
        nafill = [int(item) for item in nafill.replace(',',' ').split()]
    df_filt.fillna(nafill, inplace=True)
    return df_filt, cols

# Recode the variable if needed
def recoder(df, target):
    unique_levels = sorted([int(item) for item in df[target].unique()])
    recode_dict = {}
    while True:
        print("Enter new values for the following levels:")
        for level in unique_levels:
            new_value = input(f"Level {level}: ")
            recode_dict[level] = new_value
        
        print("\nRecode Dictionary:")
        print(recode_dict)
        check = input("Is this correct? [y/n]")
        if check.lower() == 'y' or check.lower() == 'yes':
            break
    df[target] = df[target].astype(int).map(recode_dict)
    return df

# Feature selection functions
def feature_selection_thresholding(df, tobedroppedX, target, mlfunc):
    maxfeats = int(input(f"Please specify the maximum number fo features you want: "))
    b_iteration = int(input(f"Please specify the number of iterations used for bootstrapping: "))
    stability_iter = int(input(f"Please specify the number of iterations used for feature stability testing: "))
    stability_threshold = float(input(f"Please specify the stability threshold (0-1): "))
    ths = input(f"Please specify the importance threshold(s) (0-1): ")
    ths = [float(item) for item in ths.replace(',', ' ').split()]
    def feature_selection(stability_iter, b_iteration, threshold, stability_threshold):
        feature_appearances = {}
        for stability in range(stability_iter):
            feature_importances = []
            feature_importance_means = {feature: [] for feature in X.columns}
            for _ in range(b_iteration):
                bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
                X_bootstrap = X.iloc[bootstrap_indices]
                y_bootstrap = y.iloc[bootstrap_indices]
                
                model = mlfunc(random_state=None)
                model.fit(X_bootstrap, y_bootstrap)
                feature_importances.append(model.feature_importances_)
            mean_importances = np.mean(feature_importances, axis=0) # get an average importance for each feature
            for feature, importance in zip(X.columns, mean_importances):
                feature_importance_means[feature].append(importance)
            selected_features = X.columns[mean_importances > threshold]
        
            for feature in selected_features:
                if feature in feature_appearances:
                    feature_appearances[feature] += 1
                else:
                    feature_appearances[feature] = 1
        selected_stable_features = [feature for feature, count in feature_appearances.items() 
                                    if count >= stability_threshold * stability_iter]
        stable_feature_importances = {feature: np.mean(feature_importance_means[feature]) for feature in selected_stable_features}
        sorted_features = sorted(stable_feature_importances.items(), key=lambda item: item[1], reverse=True)
        sorted_features_df = pd.DataFrame(sorted_features, columns=['Feature', 'Mean Importance'])
        
        return sorted_features_df, selected_stable_features
    
    X = df.drop(tobedroppedX, axis=1)
    y = df[target]
    thu = ths[0]
    for th in ths:
        sf_df, sf = feature_selection(stability_iter, b_iteration, th, stability_threshold)
        if len(sf) < maxfeats:
            break
        selected_features = sf
        thu = th
        print(f"Features: {sf}\nThreshold value used: {thu}")
    return selected_features, sf_df

def selectfeatures(df, tobedroppedX, target, mlfunc):
    featureselection = input("Would you like to run a feature selection? This may take significant time. [y/n]")
    while True:
        if featureselection.lower() == 'y':
            # Select features that predict meaningfully
            selected_features, sf_df = feature_selection_thresholding(df, tobedroppedX, target, mlfunc)
            break
        elif featureselection.lower() == 'n':
            while True:
                featureselection2 = input("Would you like to use all features? [y/n]")
                if featureselection2.lower() == 'y': 
                    selected_features = cols[cols not in tobedroppedX]
                    break
                elif featureselection2.lower() == 'n':
                    selected_features = input("Please list the features you wish to use: ")
                    selected_features = [item for item in selected_features.replace(',', ' ').split()]
                    break
                else:
                    print("Incorrect option selected.")
            break
        else:
            print("Incorrect option selected.")
    return selected_features

# Get information
def identify_potentials(df, cols, topamount):
    def find_closest_match(inp, opts):
        closest_match = difflib.get_close_matches(inp, opts, n=1, cutoff=0.1)
        if closest_match:
            return closest_match[0]
        else:
            return None
    
    target = input(f"Please select the target feature (options: {', '.join(cols)}): ")
    while True:
        if target not in cols:
            tguess = find_closest_match(target, cols)
            q1 = input(f"Did you mean {tguess}? [y/n]")
            if q1.lower() == 'y':
                target = tguess
                break
            elif q1.lower() == 'n':
                target = input(f"Please select the target feature: ")
            else:
                print("Incorrect option selected.")
                target = input(f"Please select the target feature: ")
        else:
            break
    targettype = input(f"Is {target}\n 1 - Categorical\n2 - Continuous")
    if targettype == '1':
        recodeq = input(f"Would you like to recode any of the levels of the target feature({target}: {sorted([int(item) for item in df[target].unique()])})? [y/n]")
        if recodeq.lower() in ('y', 'yes'):
            df = recoder(df, target)
        def mlfunc(random_state=None):
            return RandomForestClassifier(random_state=random_state)
        def paramchecker():
            return pch_RFC()
        mlname = 'RandomForestClassifier'
    elif targettype == '2':
        def mlfunc(random_state=None):
            return RandomForestRegressor(random_state=random_state)
        def paramchecker():
            return pch_RFR()
        mlname = 'RandomForestRegressor'
    else:
        print('Incorrect option selected.')
    id = input(f"Please select the ID feature, if there is one: ")
    while True:
        if id is None:
            break
        if id not in cols:
            tguess = find_closest_match(id, cols)
            q1 = input(f"Did you mean {tguess}? [y/n]")
            if q1.lower() == 'y':
                id = tguess
                break
            elif q1.lower() == 'n':
                id = input(f"Please select the ID feature: ")
            else:
                print("Incorrect option selected.")
                target = input(f"Please select the ID feature: ")
        else:
            break
    tobedroppedX = input(f"Please select the feature(s) to be removed from the training data: ")
    tobedroppedX = [item for item in tobedroppedX.replace(',', ' ').split()]
    altdroplist = []
    while True:
        for t in tobedroppedX:
            if t not in cols:
                altdroplist.append(find_closest_match(t, cols))
            else:
                altdroplist = tobedroppedX
        if altdroplist == tobedroppedX:
            break
        q1 = input(f"Did you mean {', '.join(altdroplist)}? [y/n]")
        if q1.lower() == 'y':
            tobedroppedX = altdroplist
            break
        elif q1.lower() == 'n':
            tobedroppedX = input(f"Please select the feature(s) to be removed from the training data: ")
        else:
            print("Incorrect option selected.")
            tobedroppedX = input(f"Please select the feature(s) to be removed from the training data: ")
    return df, target, tobedroppedX, id, mlname, mlfunc

# Apply model
def probabilitychecker(df, target, selected_features, n_iterations, rfm, id):
    has = df[df[target] > 0]
    no = df[df[target] == 0]
    
    probabilities = np.zeros(len(no))
    no_indices = no.index
    for _ in range(n_iterations):
        no_sample = no.sample(n=len(has), random_state=None)
        train_data = pd.concat([has, no_sample])
        
        X_train = train_data[selected_features]
        y_train = train_data[target]
        
        rfm.fit(X_train, y_train)
        y_proba = rfm.predict_proba(no[selected_features])
        probabilities += y_proba[:,1]
    
    average_probabilities = probabilities / n_iterations
    X_train = df[selected_features]
    y_train = df[target]
    
    rfm.fit(X_train, y_train)
    y_proba = rfm.predict_proba(no[selected_features])
    results = pd.DataFrame({
        id: no[id].values,
        'Average_Probability': average_probabilities
    })
    return results, rfm

# Hyperparameter checks - RandomForest Classifier
def RFC_paramgridmaker():
    nest = input("Please list the number of estimators parameters to be tested: ")
    if nest == "":
        nest = [100]
    else:
        nest = [int(item) for item in nest.replace(',', ' ').split()]
    
    
    md = input("Please list the maximum depth parameters to be tested: ")
    if md == "":
        md = [None]
    else:
        md = [int(item) for item in md.replace(',', ' ').split()]
    
    
    mss = input("Please list the minimum sample per split parameters to be tested: ")
    if mss == "":
        mss = [2]
    else:
        try:
            mss = [int(item) for item in mss.replace(',', ' ').split()]
        except ValueError:
            mss = [float(item) for item in mss.replace(',', ' ').split()]
    
    msl = input("Please list the minimum sample per leaf parameters to be tested: ")
    if msl == "":
        msl = [1]
    else:
        try:
            msl = [int(item) for item in msl.replace(',', ' ').split()]
        except ValueError:
            msl = [float(item) for item in msl.replace(',', ' ').split()]
            
    crit = input("Please specify the criterion/a (gini/entropy/log_loss/all) to be tested: ")
    if crit.lower() in ['all','a']:
        crit = ['gini', 'entropy', 'log_loss']
    else:
        crit = [item for item in crit.replace(',',' ').split()]
    
    
    mwfl = input("Please specify the minimum weighted fraction of the sum of total weights for a leaf [0.0 - 0.5]: ")
    if mwfl == "":
        mwfl = [0.0]
    else:
        mwfl = [float(item) for item in mwfl.replace(',', ' ').split()]
    
    mf = input("Please specify the maximum number of features used for a split (can be int, float or 'None', 'sqrt', 'log2'): ")
    if mf == "":
        mf = ['sqrt']
    else:
        try:
            mf = [int(item) for item in mf.replace(',', ' ').split()]
        except ValueError:
            try:
                mf = [item.lower() for item in mf.replace(',', ' ').split()]
            except ValueError:
                mf = [float(item) for item in mf.replace(',', ' ').split()]
    
    mln= input("Please specify the maximum number of leaf nodes: ")
    if mln == "":
        mln = [None]
    else:
        mln = [int(item) for item in mln.replace(',', ' ').split()]
    mid = input("Please specify the minimum impurity decrease for a node split: ")
    if mid == "":
        mid = [0.0]
    else:
        mid = [float(item) for item in mid.replace(',', ' ').split()]
    
    bs = input("Please specify if bootstrap samples are to be used [y/n]: ")
    if bs.lower() in ['true', 't', 'y', 'yes']:
        bs = True
    else:
        bs = False
    
    if bs is True:
        oobs = input("Please specify whether to use out-of-bag samples to estimate the generalization score [y/n]: ")
        if oobs.lower() in ['y', 'yes', 't', 'true']:
            oobs = True
    else:
        oobs = False
    
    ws = input("Please specify whether to resure the solution of the previous call to fit and add more estimators to ensemble [y/n]: ")
    if ws.lower() in ['true', 't', 'y', 'yes']:
        ws = True
    else:
        ws = False
    cva = int(input("Please specify the number of folds for the crossvalidation: "))

    param_grid = {
            'n_estimators': nest,
            'criterion': crit,
            'max_depth': md,
            'min_samples_split': mss,
            'min_samples_leaf': msl,
            'min_weight_fraction_leaf': mwfl,
            'max_features': mf,
            'max_leaf_nodes': mln,
            'min_impurity_decrease': mid,
            'bootstrap': [bs],
            'oob_score': [oobs],
            'max_leaf_nodes': mln,
            'warm_start': [ws]}
    return param_grid, cva

def RFC_bestparamcheck(df, tobedroppedX, target):
    selected_features = selectfeatures(df, tobedroppedX, target, mlfunc)
    scoring = {'Accuracy': make_scorer(accuracy_score)}
        
    X = df.drop(tobedroppedX, axis=1)
    y = df[target]
    while True:
        param_grid, cva = RFC_paramgridmaker()
        grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=None),
                                   param_grid=param_grid,
                                   scoring=scoring,
                                   refit='Accuracy',
                                   cv=cva,
                                   verbose=1,
                                   n_jobs = -1)
        grid_search.fit(X[selected_features], y)
        
        print("Best Parameters found by GridSearchCV:")
        print(grid_search.best_params_)
        print("Cross-Validation Results:")
        print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_Accuracy', 'std_test_Accuracy']])
        best_rf_model = grid_search.best_estimator_
        user_input = input("Would you like to proceed with these parameters? [y/n]")
        if user_input.lower() in ['y', 'yes']:
            break
    return best_rf_model, selected_features

# Hyperparameter checks - RandomForest Regressor
def RFR_paramgridmaker():
    nest = input("Please list the number of estimators parameters to be tested: ")
    if nest == "":
        nest = [100]
    else:
        nest = [int(item) for item in nest.replace(',', ' ').split()]
    
    crit = input("Please specify the criterion/a (squared_error/absolute_error/friedman_mse/poisson/all) to be tested: ")
    if crit == "":
        crit = ['squared_error']
    if crit.lower() in ['all','a']:
        crit = ['squared_error','absolute_error','friedman_mse','poisson']
    else:
        crit = [item for item in crit.replace(',',' ').split()]

    
    md = input("Please list the maximum depth parameters to be tested: ")
    if md == "":
        md = [None]
    else:
        md = [int(item) for item in md.replace(',', ' ').split()]
    
    
    mss = input("Please list the minimum sample per split parameters to be tested: ")
    if mss == "":
        mss = [2]
    else:
        try:
            mss = [int(item) for item in mss.replace(',', ' ').split()]
        except ValueError:
            mss = [float(item) for item in mss.replace(',', ' ').split()]
    
    msl = input("Please list the minimum sample per leaf parameters to be tested: ")
    if msl == "":
        msl = [1]
    else:
        try:
            msl = [int(item) for item in msl.replace(',', ' ').split()]
        except ValueError:
            msl = [float(item) for item in msl.replace(',', ' ').split()]
    
    mwfl = input("Please specify the minimum weighted fraction of the sum of total weights for a leaf [0.0 - 0.5]: ")
    if mwfl == "":
        mwfl = [0.0]
    else:
        mwfl = [float(item) for item in mwfl.replace(',', ' ').split()]
    
    mf = input("Please specify the maximum number of features used for a split (can be int, float or 'None', 'sqrt', 'log2'): ")
    if mf == "":
        mf = ['sqrt']
    else:
        try:
            mf = [int(item) for item in mf.replace(',', ' ').split()]
        except ValueError:
            try:
                mf = [item.lower() for item in mf.replace(',', ' ').split()]
            except ValueError:
                mf = [float(item) for item in mf.replace(',', ' ').split()]
    
    mln= input("Please specify the maximum number of leaf nodes: ")
    if mln == "":
        mln = [None]
    else:
        mln = [int(item) for item in mln.replace(',', ' ').split()]

    
    mid = input("Please specify the minimum impurity decrease for a node split: ")
    if mid == "":
        mid = [0.0]
    else:
        mid = [float(item) for item in mid.replace(',', ' ').split()]
    
    bs = input("Please specify if bootstrap samples are to be used [y/n]: ")
    if bs.lower() in ['true', 't', 'y', 'yes']:
        bs = True
    else:
        bs = False
    
    if bs is True:
        oobs = input("Please specify whether to use out-of-bag samples to estimate the generalization score [y/n]: ")
        if oobs.lower() in ['y', 'yes', 't', 'true']:
            oobs = True
    else:
        oobs = False
    
    ws = input("Please specify whether to resure the solution of the previous call to fit and add more estimators to ensemble [y/n]: ")
    if ws.lower() in ['true', 't', 'y', 'yes']:
        ws = True
    else:
        ws = False
    cva = int(input("Please specify the number of folds for the crossvalidation: "))

    param_grid = {
            'n_estimators': nest,
            'criterion': crit,
            'max_depth': md,
            'min_samples_split': mss,
            'min_samples_leaf': msl,
            'min_weight_fraction_leaf': mwfl,
            'max_features': mf,
            'max_leaf_nodes': mln,
            'min_impurity_decrease': mid,
            'bootstrap': [bs],
            'oob_score': [oobs],
            'max_leaf_nodes': mln,
            'warm_start': [ws]}
    return param_grid, cva

def RFR_bestparamcheck(df, tobedroppedX, target):
    selected_features = selectfeatures(df, tobedroppedX, target, mlfunc)
        
    X = df.drop(tobedroppedX, axis=1)
    y = df[target]
    while True:
        param_grid, cva = RFR_paramgridmaker()
        grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=None),
                                   param_grid=param_grid,
                                   scoring='r2',
                                   refit='Accuracy',
                                   cv=cva,
                                   verbose=1,
                                   n_jobs = -1)
        grid_search.fit(X[selected_features], y)
        
        print("Best Parameters found by GridSearchCV:")
        print(grid_search.best_params_)
        print("Cross-Validation Results:")
        print(pd.DataFrame(grid_search.cv_results_))
        best_rf_model = grid_search.best_estimator_
        user_input = input("Would you like to proceed with these parameters? [y/n]")
        if user_input.lower() in ['y', 'yes']:
            break
    return best_rf_model, selected_features