import pandas as pd
import numpy as np 
import streamlit as st

import streamlit as st
import os

# Side bar components .........................
st.sidebar.title("Projet E_comPy")
st.sidebar.header("Menu")

# Main menu status
status = st.sidebar.radio("", ("Accueil", "Introduction", "Datasets", "Preprocessing", "Data visualisation",
                               "Système de recommandation", "Modélisation", "Conclusion")
                          )

st.sidebar.image("DataScientest logo.png", width=200)

# st.sidebar.info("Projet réalisé dans le cadre de la formation **Data Analyst** de DataScientest.com \
#                Promotion **Bootcamp** octobre **2021**.")

if status == 'Accueil':
    st.title("Etude du comportement d’achat")
    st.subheader("Suivi des utilisateurs sur un site de e-commerce")
    st.image("main_photo.jpg")
    st.markdown(
        "> Projet réalisé dans le cadre de la formation **Data Analyst** de DataScientest.com Promotion **Bootcamp** octobre **2021**. ")
    # st.markdown("**Auteurs : **")
    # st.markdown("* Noor Malla")
    # st.markdown("* Denise Nguene")
    # st.markdown("* Pauline Schieven")
    # st.markdown("* Andréas Trupin")
    # st.info("Projet réalisé dans le cadre de la formation **Data Analyst** de DataScientest.com Promotion **Bootcamp** octobre **2021**.")
    st.info("**Auterus :** Noor Malla, Denise Nguene, Pauline Schieven, Andréas Trupin")


# ....................... Start of introduction section ...............................

elif status == 'Introduction':
    st.header("Introduction")
    st.markdown(
        "Le projet que nous avons réalisé est basé sur les jeux de données provenant de Kaggle (Retailrocket, e-commerce data).")
    st.markdown(
        "Les données sont brutes et ont été collectées sur un vrai site de e-commerce, pour une période de 4 mois et demi :")
    st.markdown("* Première date : 2015-05-03")
    st.markdown("* Dernière date : 2015-09-18")
    st.markdown(
        "La seule modification effectuée sur les jeux de données est le codage de certaines variables pour des raisons confidentielles.")
    st.markdown("Dans un premier temps, nous présenterons les différents jeux de données que nous avions \
                à disposition avec leurs variables. Nous explorerons ensuite les différentes tables à travers des visualisations afin de mieux comprendre les tendances et comportements du e-commerce. Par la suite, nous présenterons un système de recommandation de produits puis nous finirons par le développement de modèles de machine learning.")

# .................................. End of section .......................................

# ................................... Start of Datasets section ............................
elif status == "Datasets":
    st.header("Nos jeux de données :")
    st.subheader("1. Les actions sur le site :")
    st.markdown(
        "Pour chaque action faite sur le site, une ligne est ajoutée à la base de données  ‘events’, avec la date de l’événement et les informations de cette action. ")

    from pathlib import Path

    events = Path(__file__).parents[1] / 'DST/events.csv'
    

    st.subheader("Events")
    st.write(events)

    st.info("Cette table contient 3 types d'évenement :  **view**, **addtocart** et **transaction**.")
    st.markdown(
        "Suite à une rapide description de la table events, nous savons que : Le site est visité par **1,407,580 visiteurs** qui réalisent **2,750,455 actions** (events) sur **235,061 produits**. On peut estimer à près de 2 le nombre d'actions en moyenne par visiteur sur toute la durée du dataset, à savoir 4 mois et demi.")

    st.subheader("2. Item properties :")
    st.markdown(
        "Cette table regroupe, par date, chaque modification effectuée sur une propriété d’un item. C’est sur ces données que la majorité est modifiée pour des raisons de confidentialité, et donc cela limite fortement l’exploitation de cette table.")
    # st.image("item_properties.jpg")

    # prop_1 = load_data('item_properties_part1.csv', 10)
    # prop_2 = load_data('item_properties_part2.csv', 10)
    # properties = pd.concat([prop_1, prop_2], axis = 0)
    # st.subheader("Item_properties")
    # st.write(properties.head(5))

    # J'ai choisi que une partie comme exemplaire de la table item_properties_part1 afin d'eviter de faire "concat" online
    properties = load_data('item_properties_part1.csv', 5)
    st.subheader("properties")
    st.write(properties)

    st.subheader("3. Category tree :")
    st.markdown("La dernière table regroupe des id de catégories assignés à des id parent.")
    # st.image("categories.jpg")

    categories = load_data("category_tree.csv", 5)
    st.subheader("Categories")
    st.write(categories)


# .................................. End of section .......................................

# .................................. Start of Preprocessing section ......................
elif status == "Preprocessing":
    st.header("Preprocessing et features engineering : ")

    st.subheader("Création des variables 'nb_items_vus', 'nb_vues' et 'delta_view_add' :")
    st.markdown(
        "Pour chaque visiteurid ayant vu & ajouté au moins un produit à leur panier, nous avons construit les variables : ")
    st.markdown("- nb_items_vus : nombre d'items différents vus par le visiteurid")
    st.markdown("- nb_vues : nombre de visites sur le site (tous produits confondus) du visiteurid")
    st.markdown("- delta_view_add : le temps moyen passé (en secondes) par le visiteurid avant d'ajouter un produit à son panier")

    @st.cache
    def load_data(file_name, nrows):
        data = pd.read_csv(file_name)
        return data.head(nrows)

    predictions = load_data("predictions.csv", 100)

    st.subheader("Nouvelles variables")
    st.write(predictions)

    st.info("Cette table sera utilisée dans la partie 'Modélisation', afin de prédire si un visitorid va acheter.")


# .................................. End of section ......................................

# .................................. Start of Dataviz section ............................
elif status == "Data visualisation":
    st.title("Data visualisation : ")
    st.subheader("1. Répartition des événements : ")
    st.markdown("> **Total d'événements : 2 756 100**")
    st.image("pie_events_type.jpg", width=400)
    st.info("Nous voyons clairement que le nombre de visites représente la part la plus importante des événements du site, plus de 96%. \
            Le nombre d'achats, quant à lui, représente moins d'1% des actions totales du site. Il s'agit donc d'un site de e-commerce qui est très attractif et compte beaucoup de visites mais qui a une conversion assez faible.")

    st.subheader("2. Focus sur les produits (les top 10) : ")
    st.markdown(
        "> **L'ojectif** : observer le nombre de visites / ajouts au panier / transactions par produits et connaître les meilleurs produits pour chaque type d’action")
    st.image("top10.jpg")
    st.markdown(
        "Sur le premier graphique, on voit que le produit le plus vu n’est pas dans le top 10 des plus ajoutés au panier ou achetés. Ce produit a certainement une mauvaise conversion. Même commentaire pour les autres produits (les 3ème, 4ème ou encore 6ème produits les plus vus n'apparaissent pas dans les Top 10 suivants).")
    st.info(
        "C'est donc le deuxième item le plus vu qui est le produit le plus ajouté au panier et également le plus acheté (item 461686). De plus, ce produit n°1 compte le double d'actions que le deuxième en termes d'ajouts au panier, c'est une différence importante.")
    st.markdown(
        "Le deuxième produit le plus ajouté au panier (312728), et également le quatrième produit le plus acheté, n’est pas dans les produits les plus visités. Celui-ci est alors très acheté et mis au panier avec peu de visites au préalable. On peut donc imaginer qu’il a une très bonne conversion ou peut être un produit qui est acheté régulièrement dans le cas d’un site B2B.")
    st.markdown(
        "Les deuxième et troisième produits les plus achetés (119736 et 213834) ne sont pas parmi les plus vus et mis au panier : on peut supposer un manque de visibilité sur ces produits qui ont l’air de fonctionner à la vente. Ou également des produits qui sont régulièrement achetés directement par les entreprises (visiteurs dans le cas d’un e-commerce B2B) et qui n’ont donc pas beaucoup de visites.")

    st.markdown("> ** Le nombre de transactions par item : ** ")
    st.image("nb_transaction_par_item.jpg", width=400)
    st.info(
        "La grande majorité des produits n’ont pas été achetés sur la période étudiée. Uniquement 5 % des items ont été concernés par des transactions.")

    st.subheader("3. Focus sur les visiteurs : ")
    st.image("top30.jpg")
    st.info(
        "Nous observons que la proportion des événements du premier graphique (camembert) correspond visuellement à la proportion des événements par visiteur pour ces 30 acheteurs.")
    st.markdown(
        "> Nous pouvons en déduire que le nombre de visites et de mises au panier n'est pas parfaitement proportionnel au nombre d'achats, mais il est fortement lié. Afin de confirmer cette hypothèse, voici la heatmap de corrélation des événements ci-dessous.")
    st.image("heatmap_visitor_event.jpg", width=400)

    st.subheader("4. Nombre de vues avant l’achat : ")
    st.markdown("> ** L'objectif : savoir combien d’achats sont faits après X vues ?** ")
    st.image("nb_de_vu_avant_achat.jpg", width=400)
    st.info("La moitié des achats sont réalisés instantanément et même les ¾ sont réalisés avec une vue ou moins.")

    st.subheader("5. Evolution des événements au cours du temps : ")
    events_weekday =  pd.read_csv('events_weekday.csv')
    events_month =  pd.read_csv('events_month.csv')
    events_day =  pd.read_csv('events_day.csv')

    choix_temporalité = st.selectbox("Choisir la période à visualiser : ", ["Jour de la semaine", "Mois",
                                                               "Tous les jours"])
    if choix_temporalité == "Jour de la semaine":
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=events_weekday['weekday'],
                                 y=events_weekday['event_view'],
                                 name='Visites',
                                 line=dict(color='#66cdaa', width=2, dash='solid')))

        fig.add_trace(go.Scatter(x=events_weekday['weekday'],
                                 y=events_weekday['event_addtocart'],
                                 name='Ajouts au panier',
                                 line=dict(color='#b698f9', width=2, dash='solid')))

        fig.add_trace(go.Scatter(x=events_weekday['weekday'],
                                 y=events_weekday['event_transaction'],
                                 name='Transactions',
                                 line=dict(color='#ffa500', width=2, dash='solid')))

        fig.update_layout(title='Evenements par jour de la semaine',
                          xaxis_title='Jour de la semaine',
                          yaxis_title='Nombre d événements')

        st.plotly_chart(fig)

    elif choix_temporalité == "Mois":
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=events_month['month'],
                                 y=events_month['event_view'],
                                 name='Visites',
                                 line=dict(color='#66cdaa', width=2, dash='solid')))

        fig.add_trace(go.Scatter(x=events_month['month'],
                                 y=events_month['event_addtocart'],
                                 name='Ajouts au panier',
                                 line=dict(color='#b698f9', width=2, dash='solid')))

        fig.add_trace(go.Scatter(x=events_month['month'],
                                 y=events_month['event_transaction'],
                                 name='Transactions',
                                 line=dict(color='#ffa500', width=2, dash='solid')))

        fig.update_layout(title='Evenements par mois',
                          xaxis_title='Mois',
                          yaxis_title='Nombre d événements')
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=[5, 6, 7, 8, 9],
                ticktext=['Mai', 'Juin', 'Juillet', 'Aout', 'Septembre']
            )
        )
        st.plotly_chart(fig)

    elif choix_temporalité == "Tous les jours":

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=events_day['date'],
                             y=events_day['event_view'],
                             name='Visites',
                             line=dict(color='#66cdaa', width=2, dash='solid')))

        fig.add_trace(go.Scatter(x=events_day['date'],
                             y=events_day['event_addtocart'],
                             name='Ajouts au panier',
                             line=dict(color='#b698f9', width=2, dash='solid')))

        fig.add_trace(go.Scatter(x=events_day['date'],
                             y=events_day['event_transaction'],
                             name='Transactions',
                             line=dict(color='#ffa500', width=2, dash='solid')))

        fig.update_layout(title='Evenements par journée',
                      xaxis_title='Journée',
                      yaxis_title='Nombre d événements')

        st.plotly_chart(fig)




# .................................. End of section ......................................

# .................................. Start of sys. recommandation section .................

elif status == "Système de recommandation":
    st.header("Système de recommandation : ")
    st.subheader("Pour connaître les produits achetés par les autres visiteurs qui auraient acheté le même produit que vous")
    @st.cache
    @st.cache
    def load_data(file_name, nrows):
        data = pd.read_csv(file_name)
        return data.head(nrows)
    events = load_data("events.csv", 100000)
    acheteurs = events[events.transactionid.notnull()].visitorid.unique()
    items_achetés = []
    # Liste de tous les visiteurs qui ont acheté


    # Liste des produits achetés
    for visitor in acheteurs:
        items_achetés.append(
            list(events.loc[(events.visitorid == visitor) & (events.transactionid.notnull())].itemid.values))
    # Fonction pour identifier objets achetés avec l objet d interet
    def recommendations(item_id, items_achetés):
        # on va créer un DF avec pour index chaque produit acheté & en valeurs la liste des produits achetés avec
        liste_recommendations = []
        for x in items_achetés:
            if item_id in x:
                liste_recommendations += x
        # on retire juste le produit lui même conseillé de cette liste pour ne pas qu'il réaparaisse dans les propositions
        liste_recommendations = list(set(liste_recommendations) - set([item_id]))
        return liste_recommendations

    col_one_list = events[events.transactionid.notnull()].itemid.unique()
    col_one_list.sort()
    selectbox_01 = st.selectbox('Choisissez un itemid pour lequel vous souhaitez avoir des recommendations :', col_one_list)

    st.write("Les autres visiteurs ayant acheté l'item ", selectbox_01, ", ont également acheté les items :", recommendations(selectbox_01, items_achetés))

    if st.checkbox("Cochez cette case si vous souhaitez connaître le nombre de personne ayant déjà acheté ce produit. "):
        def nb_acheteurs(item):
            visitor_id = list(events.loc[events.itemid == item].visitorid.values)
            return len(visitor_id)
        st.write("Voici le nombre de personnes ayant déjà acheté ce produit :", nb_acheteurs(selectbox_01))

# .................................. End of section ......................................

# .................................. Start of modélisation section ........................
elif status == "Modélisation":
    st.header("Modélisation et evaluation des modèles : ")

    modele = st.selectbox("Choisir le modèle à appliquer : ", ["Clustering des produits", "Clustering des visiteurs",
                                                               "Modèle de prédiction d'achat"])
    if modele == "Clustering des produits":
        st.write("code à faire pour clustering des produits ...")

    elif modele == "Clustering des visiteurs":
        st.write("code à faire pour clustering des visiteurs ...")

    elif modele == "Modèle de prédiction d'achat":
        from sklearn import preprocessing
        from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from matplotlib import pyplot
        from sklearn import neighbors
        from sklearn import datasets

        taille=[0.2, 0.25, 0.3]
        selectbox_echantillon_test = st.selectbox('Choisissez la taille de l echantillon test :', taille)

        predictions = pd.read_csv('prediction_achat.csv', index_col=0)
        X = predictions.drop(['visitorid', 'nb_achats', 'a acheté'], axis='columns')
        y = predictions['a acheté']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=selectbox_echantillon_test)

        modele_prediction = ['LogisticRegression', 'K plus proches voisins']
        selectbox_echantillon_test = st.selectbox('Choisissez le modèle souhaité :', modele_prediction)

        if selectbox_echantillon_test == 'LogisticRegression':
            reg = LogisticRegression()
            reg.fit(X_train, y_train)

            train_acc = accuracy_score(y_true=y_train, y_pred=reg.predict(X_train))
            test_acc = accuracy_score(y_true=y_test, y_pred=reg.predict(X_test))

            st.write('- Score sur échantillon entraînement :', train_acc)
            st.write('- Score sur échantillon test :', test_acc)



        elif selectbox_echantillon_test == 'K plus proches voisins':
            st.write('Métrique imposée : minkowski')
            knn = neighbors.KNeighborsClassifier(n_neighbors=30, metric='minkowski')
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            st.write('Score sur échantillon entraînement :', knn.score(X_train, y_train))
            st.write('Score sur échantillon test :', knn.score(X_test, y_test))

# .................................. End of section ......................................

# .................................. Start of Conculsion section ........................
else:
    st.header("Conclusion ")

# .................................. End of section ......................................

