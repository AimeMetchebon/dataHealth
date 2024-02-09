import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
#from scikitplot.metrics import  plot_roc_curve, plot_precision_recall
from sklearn.metrics import precision_score, recall_score

from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)

# clf = SVC(random_state=0)
# clf.fit(X_train, y_train)
#SVC(random_state=0)
#ConfusionMatrixDisplay.from_estimator(
#----lf, X_test, y_test)
#---plt.show()


#svc_disp = RocCurveDisplay.from_estimator(svc, X_test, y_test)
#rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=svc_disp.ax_)

#clf = LogisticRegression()
#clf.fit(X_train, y_train)
#PrecisionRecallDisplay.from_estimator(clf, X_test, y_test)
def main():

    image = Image.open('banniere3.png')
    st.sidebar.image(image, caption='Evaluation de la Prédisposition à une Maladie')

    st.title("Application de Machine Learning pour déterminer la prédisposition aux maladies telles que: le diabète, les maladies cardiaques, les maladies du foie, le cancer du sein, l'insuffisance rénale"
    )
    st.subheader("Auteurs: Mireille Koutchade, Olmira Belankova et S. Aimé Metchebon T.")

    # Fonction d'importation des donnees
   
    @st.cache_data(persist=True)
    def load_datadb():
        datadb = pd.read_csv('diabete.csv')
        return datadb
    
    @st.cache_data(persist=True)
    def load_datacf():
        datacf = pd.read_csv('cancer_foie.csv')
        return datacf
    
    @st.cache_data(persist=True)
    def load_datackd():
        datackd = pd.read_csv('dfckdnn.csv')
        return datackd
    
    @st.cache_data(persist=True)
    def load_datamc():
        datamc = pd.read_csv('dfmc.csv')
        return datamc

    # Affichage de la table de donnees
    dfdb = load_datadb()
    dfdb_sample = dfdb.sample(100)

    # Affichage de la table de donnees
    dfcf = load_datacf()
    dfcf['Gender'] = dfcf['Gender'].apply(lambda x: "Female" if x == 1 else "Male")
    dfcf_sample = dfcf.sample(100)

    # Affichage de la table de donnees
    dfckd = load_datackd()
    dfckd_sample = dfckd.sample(100)

    # Affichage de la table de donnees
    dfmc = load_datamc()
    dfmc_sample = dfmc.sample(100)

    maladie = st.sidebar.radio("Afficher les donnees utulisees", ["diabete", "maladies cardiaques", "maladies du foie", "insuffisance renale"])
    if  maladie == "diabete":
        st.subheader("Jeu de donnees 'Diabete': Echantillon de 100 observations")
        st.write(dfdb_sample)

        #st.write(dfdb_sample)
        seed = 123

        # train/test split
        @st.cache_data(persist=True)
        def split(df):
            y = df.iloc[:,-1]
            X = df.drop(df.columns[-1], axis = 1)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
            test_size= 0.25,
            stratify=y, # permet de faire respecter les proportions dans les jeu d'entrainement et de de test
            random_state = seed
            
            )
            return X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = split(dfdb)

       
        classifier = st.sidebar.selectbox(
            "Classificateur",
            ("Random Forest", "K-Nearest Neighbours", "SVM","Decision Tree","Logistic Regression")

        )

        
        # Analyse de la performance des modeles
        def plot_perf(graphes):
            if 'Confusion matrix' in graphes:
                st.subheader('Matrice de confusion')
                ConfusionMatrixDisplay.from_estimator(
                model, X_test, y_test
                )
                st.pyplot()

            if 'ROC curve' in graphes:
                st.subheader('Courbe ROC')
                RocCurveDisplay.from_estimator(
                model, X_test, y_test
                 )
                st.pyplot()

            if 'Precision-Recall curve' in graphes:
                st.subheader('Precision-Recall curve')
                PrecisionRecallDisplay.from_estimator(
                 model, X_test, y_test
                )
                st.pyplot()

        graphes_perf = st.sidebar.multiselect(
        "Choisir un graphique de performance du modele ML",
        ("Confusion matrix", "ROC curve","Precision-Recall curve")
        )

        #l'utilisateur saisie une valeur pour chaque parametre du patient
        st.subheader("Donnees pour la prediction de la predisposition a la maladie")
        Pregnancies = st.number_input(label='Nombre de Grossesse', min_value=0, value = 4)
        Glucose = st.number_input(label='Taux de glucose', min_value=0, value = 93)
        BloodPressure = st.number_input(label='Pression du sang', min_value=0, value = 68)
        SkinThickness = st.number_input(label='epaisseur de la peau', min_value=0, value = 25)
        Insulin = st.number_input(label="taux d'insuline", min_value=0, value = 93)
        BMI = st.number_input(label="BMI", min_value=0., value = 31.1)
        DiabetesPedigreeFunction = st.number_input(label="Antecedent familiaux", min_value=0.000, value = 0.183)
        Age = st.number_input(label="Age", min_value=0, value = 26)

         

        # Random Forest
        if classifier == "Random Forest" :
            st.sidebar.subheader("Hyperparamètre du modele")
            n_arbres = st.sidebar.number_input(
            "Choisir le nombre d'arbre dans la forêt",
            100, 1000, step =10
            )
            profondeur_arbre = st.sidebar.number_input(
            "Choisir la profondeur maximale d'un arbre", 1, 20, step=1
            )
            bootstrap = st.sidebar.radio(
            "Echantillons bootstrap lors de la creation d'arbre ?",
            ("True", "False")
            )

        
            def inference(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
                new_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                BMI, DiabetesPedigreeFunction, Age]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classify"):
                st.subheader("Random Forest Results")
        
                # initiatisation d'un objet RandomForestClassifier
                model = RandomForestClassifier(
                n_estimators = n_arbres,
                max_depth= profondeur_arbre,
                bootstrap= bool(bootstrap)
                )
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
                if prediction[0]== 1:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé au diabète"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé au diabète"

                st.write(resultat)

        # KK-Nearest Neighbours
        if classifier == "K-Nearest Neighbours" :
            st.sidebar.subheader("Hyperparamètre du modele")
            n_voisin = st.sidebar.number_input(
            "Choisir le nombre de voisins",
            3, 10, step =1
            )
            
        
            def inference(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
                new_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                BMI, DiabetesPedigreeFunction, Age]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classify"):
                st.subheader("K-Nearest Neighbours Results")
        
                # initiatisation d'un objet RandomForestClassifier
                model = KNeighborsClassifier(
                n_neighbors = n_voisin
                
                )
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
                if prediction[0]== 1:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé au diabète"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé au diabète"

                st.write(resultat)
        # SVM
        if classifier == "SVM" :
                    
            def inference(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
                new_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                BMI, DiabetesPedigreeFunction, Age]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classify"):
                st.subheader("SVM Results")
        
                # initiatisation d'un objet RandomForestClassifier
                model = SVC()
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
                if prediction[0]== 1:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé au diabète"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé au diabète"

                st.write(resultat)
    
        # "Decision Tree"
        if classifier == "Decision Tree" :
            st.sidebar.subheader("Hyperparamètre du modele")
            profondeur_max = st.sidebar.number_input(
            "Choisir une profondeur maximale",
            1, 25, step =1
            )
            n_echan_min = st.sidebar.number_input(
            "Choisir le nombre minimal d'echantillon dans un noeud", 1, 5, step=1
            )
            n_echan_min_div = st.sidebar.number_input(
            "Choisir le nombre minimal d'echantillon pour diviser un noeud", 2, 10, step=3
            )
        
            def inference(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
                new_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                BMI, DiabetesPedigreeFunction, Age]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classify"):
                st.subheader("Decision Tree Results")
        
                # initiatisation d'un objet RandomForestClassifier
                model = DecisionTreeClassifier(
                max_depth = profondeur_max,
                min_samples_leaf= n_echan_min,
                min_samples_split= n_echan_min_div
                )
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
                if prediction[0]== 1:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé au diabète"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé au diabète"

                st.write(resultat)
        
        # Logistic Regression
        if classifier == "Logistic Regression" :
                    
            def inference(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
                new_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                BMI, DiabetesPedigreeFunction, Age]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classify"):
                st.subheader("Logistic Regression Results")
        
                # initiatisation d'un objet RandomForestClassifier
                model = LogisticRegression()
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
                if prediction[0]== 1:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé au diabète"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé au diabète"

                st.write(resultat)

    
    if maladie == "maladies du foie":
        st.subheader("Jeu de donnees 'maladies du foie': Echantillon de 100 observations")
        st.write(dfcf_sample)

        #st.write(dfdb_sample)
        seed = 32

        # train/test split
        @st.cache_data(persist=True)
        def split(df):
            y = df.iloc[:,-1]
            X = df.drop(df.columns[-1], axis = 1)
            X['Gender']=X['Gender'].apply(lambda x: 1 if x =='Female' else 2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
            test_size= 0.25,
            stratify=y, # permet de faire respecter les proportions dans les jeu d'entrainement et de de test
            random_state = seed
            
            )
            return X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = split(dfcf)

       
        classifier = st.sidebar.selectbox(
            "Classificateur",
            ("Random Forest", "K-Nearest Neighbours", "SVM","Decision Tree","Logistic Regression")

        )

        
        # Analyse de la performance des modeles
        def plot_perf(graphes):
            if 'Confusion matrix' in graphes:
                st.subheader('Matrice de confusion')
                ConfusionMatrixDisplay.from_estimator(
                model, X_test, y_test
                )
                st.pyplot()

            if 'ROC curve' in graphes:
                st.subheader('Courbe ROC')
                RocCurveDisplay.from_estimator(
                model, X_test, y_test
                 )
                st.pyplot()

            if 'Precision-Recall curve' in graphes:
                st.subheader('Precision-Recall curve')
                PrecisionRecallDisplay.from_estimator(
                 model, X_test, y_test
                )
                st.pyplot()

        graphes_perf = st.sidebar.multiselect(
        "Choisir un graphique de performance du modele ML",
        ("Confusion matrix", "ROC curve","Precision-Recall curve")
        )

        #l'utilisateur saisie une valeur pour chaque parametre du patient
        st.subheader("Donnees pour la prediction de la predisposition au cancer du foie")
        Age = st.number_input(label='Age', min_value=0, value = 45)

        Gender = st.selectbox("Genre", ("Female", "Male"))
        #Gender = st.text_input('Genre', 'Female')
        if Gender == "Female":
            Gender = '1'
        else:
            Gender = '2'
        Gender = int(Gender)
        Total_Bilirubin = st.number_input(label='Total_Bilirubin', min_value=0., value = 1.0)
        Alkaline_Phosphotase = st.number_input(label='Alkaline_Phosphotase', min_value=0., value = 208.0)
        Alamine_Aminotransferase = st.number_input(label="Alamine_Aminotransferase", min_value=0., value = 35.0)
        Albumin_and_Globulin_Ratio = st.number_input(label="Albumin_and_Globulin_Ratio", min_value=0., value = 0.93)
        

        # Random Forest
        if classifier == "Random Forest" :
            st.sidebar.subheader("Hyperparamètre du modele")
            n_arbres = st.sidebar.number_input(
            "Choisir le nombre d'arbre dans la forêt",
            100, 1000, step =10
            )
            profondeur_arbre = st.sidebar.number_input(
            "Choisir la profondeur maximale d'un arbre", 1, 20, step=1
            )
            bootstrap = st.sidebar.radio(
            "Echantillons bootstrap lors de la creation d'arbre ?",
            ("True", "False")
            )

        
            def inference(Age, Gender, Total_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Albumin_and_Globulin_Ratio):
                new_data = np.array([Age, Gender, Total_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Albumin_and_Globulin_Ratio]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classify"):
                st.subheader("Random Forest Results")
        
                # initiatisation d'un objet RandomForestClassifier
                model = RandomForestClassifier(
                n_estimators = n_arbres,
                max_depth= profondeur_arbre,
                bootstrap= bool(bootstrap)
                )
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(Age, Gender, Total_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Albumin_and_Globulin_Ratio)
                if prediction[0]== 2:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé au cancer du foie"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé au cancer du foie"

                st.write(resultat)

        # KK-Nearest Neighbours
        if classifier == "K-Nearest Neighbours" :
            st.sidebar.subheader("Hyperparamètre du modele")
            n_voisin = st.sidebar.number_input(
            "Choisir le nombre de voisins",
            3, 10, step =1
            )
            
        
            def inference(Age, Gender, Total_Bilirubin, Alkaline_Phosphotase,Alamine_Aminotransferase, Albumin_and_Globulin_Ratio):
                new_data = np.array([Age, Gender,Total_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Albumin_and_Globulin_Ratio]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classify"):
                st.subheader("K-Nearest Neighbours Results")
        
                # initiatisation d'un objet RandomForestClassifier
                model = KNeighborsClassifier(
                n_neighbors = n_voisin
                
                )
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(Age, Gender, Total_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Albumin_and_Globulin_Ratio)
                if prediction[0]== 2:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé au cancer du foie"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé au cancer du foie"

                st.write(resultat)
        # SVM
        if classifier == "SVM" :
                    
            def inference(Age, Gender, Total_Bilirubin, Alkaline_Phosphotase,Alamine_Aminotransferase, Albumin_and_Globulin_Ratio):
                new_data = np.array([Age, Gender, Total_Bilirubin, Alkaline_Phosphotase,Alamine_Aminotransferase, Albumin_and_Globulin_Ratio]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classify"):
                st.subheader("SVM Results")
        
                # initiatisation d'un objet RandomForestClassifier
                model = SVC()
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(Age, Gender, Total_Bilirubin, Alkaline_Phosphotase,Alamine_Aminotransferase, Albumin_and_Globulin_Ratio)
                if prediction[0]== 2:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé au cancer du foie"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé au cancer du foie"

                st.write(resultat)
    
        # "Decision Tree"
        if classifier == "Decision Tree" :
            st.sidebar.subheader("Hyperparamètre du modele")
            profondeur_max = st.sidebar.number_input(
            "Choisir une profondeur maximale",
            1, 25, step =1
            )
            n_echan_min = st.sidebar.number_input(
            "Choisir le nombre minimal d'echantillon dans un noeud", 1, 5, step=1
            )
            n_echan_min_div = st.sidebar.number_input(
            "Choisir le nombre minimal d'echantillon pour diviser un noeud", 2, 10, step=3
            )
        
            def inference(Age, Gender, Total_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Albumin_and_Globulin_Ratio):
                new_data = np.array([Age, Gender, Total_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Albumin_and_Globulin_Ratio]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classify"):
                st.subheader("Decision Tree Results")
        
                # initiatisation d'un objet RandomForestClassifier
                model = DecisionTreeClassifier(
                max_depth = profondeur_max,
                min_samples_leaf= n_echan_min,
                min_samples_split= n_echan_min_div
                )
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(Age, Gender, Total_Bilirubin, Alkaline_Phosphotase,Alamine_Aminotransferase, Albumin_and_Globulin_Ratio)
                if prediction[0]== 2:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé au cancer du foie"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé au cancer du foie"

                st.write(resultat)
        
        # Logistic Regression
        if classifier == "Logistic Regression" :
                    
            def inference(Age, Gender, Total_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Albumin_and_Globulin_Ratio):
                new_data = np.array([Age, Gender, Total_Bilirubin, Alkaline_Phosphotase,Alamine_Aminotransferase, Albumin_and_Globulin_Ratio]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classifycf"):
                st.subheader("Logistic Regression Results")
        
                # initiatisation d'un objet RandomForestClassifier
                model = LogisticRegression()
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(Age, Gender, Total_Bilirubin, Alkaline_Phosphotase,Alamine_Aminotransferase, Albumin_and_Globulin_Ratio)
                if prediction[0]== 2:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé au cancer du foie"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé au cancer du foie"

                st.write(resultat)

       
    if maladie == "insuffisance renale":
        st.subheader("Jeu de donnees 'insuffisance renale': Echantillon de 100 observations")
        st.write(dfckd_sample)

        st.write(dfdb_sample)
        seed = 30

        #train/test split
        @st.cache_data(persist=True)
        def split(df):
            y = df.iloc[:,-1]
            X = df.drop(df.columns[-1], axis = 1)
            X_train, X_test, y_train, y_test = train_test_split(
               X, y, 
           test_size= 0.25,
            stratify=y, # permet de faire respecter les proportions dans les jeu d'entrainement et de de test
            random_state = seed
            
            )
            return X_train, X_test, y_train, y_test
        
        dfckd['rbc']= dfckd['rbc'].apply(lambda x: 0 if x == "normal" else 1 )
        dfckd['pc']= dfckd['pc'].apply(lambda x: 0 if x == "normal" else 1 )
        dfckd['pcc']= dfckd['pcc'].apply(lambda x: 0 if x == "notpresent" else 1 )
        dfckd['ba']= dfckd['ba'].apply(lambda x: 0 if x == "notpresent" else 1 )
        dfckd['htn']= dfckd['htn'].apply(lambda x: 0 if x == "no" else 1 )
        dfckd['dm']= dfckd['dm'].apply(lambda x: 0 if x == "no" else 1 )
        dfckd['cad']= dfckd['cad'].apply(lambda x: 0 if x == "no" else 1 )
        dfckd['appet']= dfckd['appet'].apply(lambda x: 0 if x == "good" else 1 )
        dfckd['pe']= dfckd['pe'].apply(lambda x: 0 if x == "no" else 1 )
        dfckd['ane']= dfckd['ane'].apply(lambda x: 0 if x == "no" else 1 ) 
        dfckd['classification']= dfckd['classification'].apply(lambda x: 1 if x == "ckd" else 0 )

        X_train, X_test, y_train, y_test = split(dfckd)
        
       
        classifier = st.sidebar.selectbox(
            "Classificateur",
            ("Random Forest", "K-Nearest Neighbours", "SVM","Decision Tree","Logistic Regression")

        )

        
        # Analyse de la performance des modeles
        def plot_perf(graphes):
            if 'Confusion matrix' in graphes:
                st.subheader('Matrice de confusion')
                ConfusionMatrixDisplay.from_estimator(
                model, X_test, y_test
                )
                st.pyplot()

            if 'ROC curve' in graphes:
                st.subheader('Courbe ROC')
                RocCurveDisplay.from_estimator(
                model, X_test, y_test
                 )
                st.pyplot()

            if 'Precision-Recall curve' in graphes:
                st.subheader('Precision-Recall curve')
                PrecisionRecallDisplay.from_estimator(
                 model, X_test, y_test
                )
                st.pyplot()

        graphes_perf = st.sidebar.multiselect(
        "Choisir un graphique de performance du modele ML",
        ("Confusion matrix", "ROC curve","Precision-Recall curve")
        )

        #l'utilisateur saisie une valeur pour chaque parametre du patient
        st.subheader("Donnees pour la prediction de la predisposition aux maladies du foie")
        age = st.number_input(label='age', min_value=0, value = 55)
        bp = st.number_input(label='Pression arterielle (bp)', min_value=0., value = 75.7)
        sg = st.number_input(label="garvité spécifiaue de l'urine(sg)", min_value=0., value = 1.01)
        al = st.number_input(label='Taux albumine dans le sang(al)', min_value=0., value = 0.90)
        su = st.number_input(label="Taux de sucre dans le sang(su)", min_value=0., value = 0.38)

        #rbc = st.text_input("Nombre de globule rouge(rbc)", "normal")
        rbc = st.selectbox("Nombre de globule rouge(rbc)", ("normal", "abnormal"))
        if rbc == "normal":
            rbc = '0'
        else:
            rbc = '1'
        rbc = int(rbc)

        pc = st.selectbox("Presence de cellule de pus (pc)", ("normal", "abnormal"))
        #pc = st.text_input("Presence de cellule de pus (pc)", "normal")
        if pc == "normal":
            pc = '0'
        else:
            pc = '1'
        pc = int(pc)
        
        pcc = st.selectbox("Agregats de cellule de pus (pcc)", ("present", "notpresent"))
        #pcc = st.text_input("Agregats de cellule de pus (pcc)", "notpresent")
        if pcc == "notpresent":
            pcc = '0'
        else:
            pcc = '1'
        pcc = int(pcc)

        ba = st.selectbox("Présence de bacteries (ba)", ("present", "notpresent"))
        #ba = st.text_input("Présence de bacteries (ba)", "present")
        if ba == "notpresent":
            ba = '0'
        else:
            ba = '1'
        ba = int(ba)
        bgr = st.number_input(label="Taux de sucre sanguin aléqtoire(bgr)", min_value=0., value = 143.7)
        bu = st.number_input(label="Taux d'urée sanguin(bu)", min_value=0., value = 54.90)
        sc = st.number_input(label="Taux de créatine sérique(sc)", min_value=0., value = 2.52)
        sod = st.number_input(label="Taux de sodium (sod)", min_value=0., value = 137.98)
        pot = st.number_input(label="Taux de potassium (pot)", min_value=0., value = 4.33)
        hemo = st.number_input(label="Taux d'hémoglobine (hemo)", min_value=0., value = 12.98)
        pcv = st.number_input(label="Volume de globules rouges tassé (pcv)", min_value=0., value = 39.64)
        wc = st.number_input(label="Nombre de globules blanc (wc)", min_value=0., value = 8438.09)
        rc = st.number_input(label="Nombre de globules rouges (rc)", min_value=0., value = 4.70)
        
        htn = st.selectbox("Présence d'hypertension (htn)", ("yes", "no"))
        #htn = st.text_input("Présence d'hypertension (htn)", "no")
        if htn == "no":
            htn = '0'
        else:
            htn = '1'
        htn = int(htn)
        
        dm = st.selectbox("Présence de diabète sucré (dm)", ("yes", "no"))
        #dm = st.text_input("Présence de diabète sucré (dm)", "yes")
        if dm == "no":
            dm = '0'
        else:
            dm = '1'
        dm = int(dm)

        cad = st.selectbox("Présence de bacteries (cad)", ("yes", "no"))
        #cad = st.text_input("Présence de bacteries (cad)", "no")
        if cad == "no":
            cad = '0'
        else:
            cad = '1'
        cad = int(cad)

        appet = st.selectbox( "Niveau d'appétit  (appet)", ("good", "poor"))
        #appet = st.text_input("Niveau d'appétit  (appet)", "good")
        if appet == "good":
            appet = '0'
        else:
            appet = '1'
        appet = int(appet)
        pe = st.selectbox( "Présence d'oedème des pieds (pe)", ("yes", "no"))
        #pe = st.text_input("Présence d'oedème des pieds (pe)", "yes")
        if pe == "yes":
            pe = '0'
        else:
            pe = '1'
        pe = int(pe)

        ane = st.selectbox("Présence d'anémie (ane)", ("yes", "no"))
        #ane = st.text_input("Présence d'anémie (ane)", "no")
        if ane == "no":
            ane = '0'
        else:
            ane = '1'
        ane = int(ane)


        
        # Random Forest
        if classifier == "Random Forest" :
            st.sidebar.subheader("Hyperparamètre du modele")
            n_arbres = st.sidebar.number_input(
            "Choisir le nombre d'arbre dans la forêt",
            100, 1000, step =10
            )
            profondeur_arbre = st.sidebar.number_input(
            "Choisir la profondeur maximale d'un arbre", 1, 20, step=1
            )
            bootstrap = st.sidebar.radio(
            "Echantillons bootstrap lors de la creation d'arbre ?",
            ("True", "False")
            )

        
            def inference(age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu,
                sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad,appet, pe, ane):
                new_data = np.array([age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu,
                sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad,appet, pe, ane]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classify"):
                st.subheader("Random Forest Results")
        
                # initiatisation d'un objet RandomForestClassifier
                model = RandomForestClassifier(
                n_estimators = n_arbres,
                max_depth= profondeur_arbre,
                bootstrap= bool(bootstrap)
                )
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu,
                sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad,appet, pe, ane)
                if prediction[0]== 1:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé à une insuffisance rénale"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé à une insuffisance rénale"

                st.write(resultat)

        # KK-Nearest Neighbours
        if classifier == "K-Nearest Neighbours" :
            st.sidebar.subheader("Hyperparamètre du modele")
            n_voisin = st.sidebar.number_input(
            "Choisir le nombre de voisins",
            3, 10, step =1
            )
            
        
            def inference(age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu,
                sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad,appet, pe, ane):
                new_data = np.array([age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu,
                sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad,appet, pe, ane]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classify"):
                st.subheader("RK-Nearest Neighbours")
        
                # initiatisation d'un objet RandomForestClassifier
                model = KNeighborsClassifier(
                n_neighbors = n_voisin
                
                )
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu,
                sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad,appet, pe, ane)
                if prediction[0]== 1:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé à une insuffisance rénale"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé à une insuffisance rénale"

                st.write(resultat)
        # SVM
        if classifier == "SVM" :
                    
            def inference(age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu,
                sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad,appet, pe, ane):
                new_data = np.array([age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu,
                sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad,appet, pe, ane]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classify"):
                st.subheader("SVM")
        
                # initiatisation d'un objet RandomForestClassifier
                model = SVC()
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu,
                sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad,appet, pe, ane)
                if prediction[0]== 1:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé à une insuffisance rénale"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé à une insuffisance rénale"

                st.write(resultat)
    
        # "Decision Tree"
        if classifier == "Decision Tree" :
            st.sidebar.subheader("Hyperparamètre du modele")
            profondeur_max = st.sidebar.number_input(
            "Choisir une profondeur maximale",
            1, 25, step =1
            )
            n_echan_min = st.sidebar.number_input(
            "Choisir le nombre minimal d'echantillon dans un noeud", 1, 5, step=1
            )
            n_echan_min_div = st.sidebar.number_input(
            "Choisir le nombre minimal d'echantillon pour diviser un noeud", 2, 10, step=3
            )
        
            def inference(age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu,
                sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad,appet, pe, ane):
                new_data = np.array([age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu,
                sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad,appet, pe, ane]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classify"):
                st.subheader("Decision Tree")
        
                # initiatisation d'un objet RandomForestClassifier
                model = DecisionTreeClassifier(
                max_depth = profondeur_max,
                min_samples_leaf= n_echan_min,
                min_samples_split= n_echan_min_div
                )
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu,
                sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad,appet, pe, ane)
                if prediction[0]== 1 :
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé à une insuffisance rénale"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé à une insuffisance rénale"

                st.write(resultat)
        
        # Logistic Regression
        if classifier == "Logistic Regression" :
                    
            def inference(age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu,
                sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad,appet, pe, ane):
                new_data = np.array([age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu,
                sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad,appet, pe, ane]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classifycf"):
                st.subheader("Logistic Regression")
        
                # initiatisation d'un objet RandomForestClassifier
                model = LogisticRegression()
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu,
                sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad,appet, pe, ane)
                if prediction[0]== 1:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé à une insuffisance rénale"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé à une insuffisance rénale"

                st.write(resultat)

    if maladie == "maladies cardiaques":
        st.subheader("Jeu de donnees 'maladies cardiaques': Echantillon de 100 observations")
        st.write(dfmc_sample)

        seed = 30

        #train/test split
        @st.cache_data(persist=True)
        def split(df):
            y = df.iloc[:,-1]
            X = df.drop(df.columns[-1], axis = 1)
            X_train, X_test, y_train, y_test = train_test_split(
               X, y, 
           test_size= 0.25,
            stratify=y, # permet de faire respecter les proportions dans les jeu d'entrainement et de de test
            random_state = seed
            
            )
            return X_train, X_test, y_train, y_test
        
        
        X_train, X_test, y_train, y_test = split(dfmc)
        
       
        classifier = st.sidebar.selectbox(
            "Classificateur",
            ("Random Forest", "K-Nearest Neighbours", "SVM","Decision Tree","Logistic Regression")

        )

        
        # Analyse de la performance des modeles
        def plot_perf(graphes):
            if 'Confusion matrix' in graphes:
                st.subheader('Matrice de confusion')
                ConfusionMatrixDisplay.from_estimator(
                model, X_test, y_test
                )
                st.pyplot()

            if 'ROC curve' in graphes:
                st.subheader('Courbe ROC')
                RocCurveDisplay.from_estimator(
                model, X_test, y_test
                 )
                st.pyplot()

            if 'Precision-Recall curve' in graphes:
                st.subheader('Precision-Recall curve')
                PrecisionRecallDisplay.from_estimator(
                 model, X_test, y_test
                )
                st.pyplot()

        graphes_perf = st.sidebar.multiselect(
        "Choisir un graphique de performance du modele ML",
        ("Confusion matrix", "ROC curve","Precision-Recall curve")
        )

        #l'utilisateur saisie une valeur pour chaque parametre du patient
        st.subheader("Donnees pour la prediction de la predisposition aux maladies du foie")
        age = st.number_input(label='age', min_value=0., value = 67.0)

        sex = st.selectbox(" Your sex", ("1: Male","0: Female"))
        if sex == "1: Male":
            sex = '1'
        else:
            sex = '0'
        sex = int(sex)

        cp = st.selectbox(" chest pain type (cp)", ("1: atypical angina","0: symptomatic", "2:non-anginal","3:typical angina"))
        if cp == "1: atypical angina":
            cp = '1'
        elif cp == "0: symptomatic":
            cp = '0'
        elif cp == "2:non-anginal":
            cp = '2'
        else:
            cp = '3'
        cp = int(cp)
        trestbps = st.number_input(label="Person's resting blood pressure (trestbps)", min_value=0., value = 160.0)
        chol = st.number_input(label="Person's cholesterol measurement (chol)", min_value=0., value = 244.0)
        fbs = st.number_input(label="Preson's fasting blood sugar(fbs)", min_value=0., value = 0.0)
        #restecg = st.number_input(label="Resting electrocardiographic results (restecg)", min_value=0., value = 0.38)
        restecg = st.selectbox("Resting electrocardiographic results (restecg)", ("1:normal","0:probale or definite left ventricular hypertrophy","2:having ST-T wave abnormality"))
        if restecg  == "1:normal" :
            restecg = '1'
        elif restecg == "0:probable or definite left ventricular hypertrophy":
            restecg = '0'
        else:
            restecg = '2'
        restecg = int(restecg)
        
        thalach = st.number_input(label="Person's maximum heart rate achieved (thalach)", min_value=0., value = 118.0)
        #exang = st.number_input(label="Exercise induced angina (exang)", min_value=0., value = 0.38)

        exang = st.selectbox("Exercise induced angina (exang)", ("1: yes","0: no"))
        if exang == "1: yes":
            exang = '1'
        else:
            exang = '0'
        exang = int(exang)

        oldpeak = st.number_input(label="ST depression induced by exercise relative to rest (oldpeak)", min_value=0., value = 0.0)
        
        slope = st.selectbox("the slope of the peak exercise ST segment (slope)", ("1:flat","0:downsloping","2:upsloping"))
        if slope  == "1:flat"  :
            slope = '1'
        elif slope == "0:downsloping":
            slope = '0'
        else:
            slope = '2'
        slope = int(slope)

        ca = st.number_input(label="Number of major vessels (ca)", min_value=0., value = 3.0)
        
        thal = st.selectbox("Blood disorder called thalassemia (thal)", ("1:fixed defect", "2:normal blood flow","3:reversible defect"))
        if thal == "1:fixed defect":
            thal = '1'
        elif thal == "2:normal blood flow":
            thal = '2'
        else:
            thal = '3'
        thal = int(thal)
        

       
        # Random Forest
        if classifier == "Random Forest" :
            st.sidebar.subheader("Hyperparamètre du modele")
            n_arbres = st.sidebar.number_input(
            "Choisir le nombre d'arbre dans la forêt",
            100, 1000, step =10
            )
            profondeur_arbre = st.sidebar.number_input(
            "Choisir la profondeur maximale d'un arbre", 1, 20, step=1
            )
            bootstrap = st.sidebar.radio(
            "Echantillons bootstrap lors de la creation d'arbre ?",
            ("True", "False")
            )

        
            def inference(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal):
                new_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classify"):
                st.subheader("Random Forest Results")
        
                # initiatisation d'un objet RandomForestClassifier
                model = RandomForestClassifier(
                n_estimators = n_arbres,
                max_depth= profondeur_arbre,
                bootstrap= bool(bootstrap)
                )
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal)
                if prediction[0]== 1:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé aux maladie cardiaques"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé aux maladies cardiaques"

                st.write(resultat)

        # KK-Nearest Neighbours
        if classifier == "K-Nearest Neighbours" :
            st.sidebar.subheader("Hyperparamètre du modele")
            n_voisin = st.sidebar.number_input(
            "Choisir le nombre de voisins",
            3, 10, step =1
            )
            
        
            def inference(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal):
                new_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classify"):
                st.subheader("RK-Nearest Neighbours")
        
                # initiatisation d'un objet RandomForestClassifier
                model = KNeighborsClassifier(
                n_neighbors = n_voisin
                
                )
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal)
                if prediction[0]== 1:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé aux maladies cardiaques"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé aux maladies cardiaques"

                st.write(resultat)
        # SVM
        if classifier == "SVM" :
                    
            def inference(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal):
                new_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classify"):
                st.subheader("SVM")
        
                # initiatisation d'un objet RandomForestClassifier
                model = SVC()
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal)
                if prediction[0]== 1:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé aux maladies cardiaques"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé aux maladies cardiaques"

                st.write(resultat)
    
        # "Decision Tree"
        if classifier == "Decision Tree" :
            st.sidebar.subheader("Hyperparamètre du modele")
            profondeur_max = st.sidebar.number_input(
            "Choisir une profondeur maximale",
            1, 25, step =1
            )
            n_echan_min = st.sidebar.number_input(
            "Choisir le nombre minimal d'echantillon dans un noeud", 1, 5, step=1
            )
            n_echan_min_div = st.sidebar.number_input(
            "Choisir le nombre minimal d'echantillon pour diviser un noeud", 2, 10, step=3
            )
        
            def inference(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal):
                new_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classify"):
                st.subheader("Decision Tree")
        
                # initiatisation d'un objet RandomForestClassifier
                model = DecisionTreeClassifier(
                max_depth = profondeur_max,
                min_samples_leaf= n_echan_min,
                min_samples_split= n_echan_min_div
                )
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal)
                if prediction[0]== 1 :
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé aux maladies cardiaques"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé aux maladies cardiaques"

                st.write(resultat)
        
        # Logistic Regression
        if classifier == "Logistic Regression" :
                    
            def inference(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal):
                new_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal]
                )
                pred = model.predict(new_data.reshape(1,-1))
                return pred

            if st.sidebar.button("Execution", key = "classifycf"):
                st.subheader("Logistic Regression")
        
                # initiatisation d'un objet RandomForestClassifier
                model = LogisticRegression()
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)


                # Metriques de performance
                accuracy = model.score(X_test, y_test)
                accuracy_on_the_train_set = model.score(X_train, y_train)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application
                st.write("Accuracy:", round(accuracy,3))
                st.write("Accuracy_on_the_train_set:", round(accuracy_on_the_train_set,3))
                st.write("Precision:", round(precision,3))
                st.write("Recall", round(recall,3))

                # Afficher les graphiques de performance
                plot_perf(graphes_perf)   
        
        
                st.subheader("Prediction de la predisposition a la maladie")
                prediction = inference(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal)
                if prediction[0]== 1:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient est predisposé aux maladies cardiaques"
                else:
                    resultat = f"la valeur de la classe du patient est {prediction[0]} : patient n'est pas predisposé aux maladies cardiaques"

                st.write(resultat)


    


if __name__ == '__main__':
    main()





