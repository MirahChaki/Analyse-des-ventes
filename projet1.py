
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# Chemin vers le fichier CSV
file_path = '/Users/chakimirah/Desktop/M2 IMDS/Projet personel/Analyse des ventes d\'un magasin/sales_data_sample.csv'

# Charger les données avec un encodage spécifique
df = pd.read_csv(file_path, encoding='ISO-8859-1') 

# Afficher les colonnes pour vérifier les noms
print(df.columns)

# Afficher les premières lignes pour vérifier que les données sont chargées
print(df.head())

# Informations sur le DataFrame
print(df.info())

# Statistiques descriptives
print(df.describe())

# Connexion à une base de données SQLite
conn = sqlite3.connect('company_performance.db')

# Écrire le DataFrame dans une table SQL
df.to_sql('performance', conn, if_exists='replace', index=False)

# Vérifier la table
print(pd.read_sql_query("SELECT * FROM performance LIMIT 5;", conn))

# Total des ventes par ligne de produit
total_sales_per_product = df.groupby('PRODUCTLINE')['SALES'].sum().reset_index()  # Utilise 'PRODUCTLINE' et 'SALES'
print(total_sales_per_product.sort_values(by='SALES', ascending=False))

# Histogramme des ventes
plt.hist(df['SALES'], bins=20, color='skyblue', edgecolor='black')  # Utilise 'SALES' pour l'histogramme
plt.title('Distribution des ventes')
plt.xlabel('Montant des ventes')
plt.ylabel('Fréquence')
plt.grid(axis='y')
plt.show()

# Analyse temporelle
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])  # Convertir la colonne en datetime
monthly_sales = df.resample('M', on='ORDERDATE')['SALES'].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['ORDERDATE'], monthly_sales['SALES'], marker='o')
plt.title('Ventes au fil du temps')
plt.xlabel('Date')
plt.ylabel('Ventes Totales')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Analyse par région
sales_by_region = df.groupby('COUNTRY')['SALES'].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.bar(sales_by_region['COUNTRY'], sales_by_region['SALES'], color='skyblue')
plt.title('Ventes Totales par Pays')
plt.xlabel('Pays')
plt.ylabel('Ventes Totales')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Analyse par produit
top_products = df.groupby('PRODUCTLINE')['SALES'].sum().reset_index()
top_products = top_products.sort_values(by='SALES', ascending=False).head(10)  # Top 10

plt.figure(figsize=(12, 6))
plt.bar(top_products['PRODUCTLINE'], top_products['SALES'], color='lightgreen')
plt.title('Top 10 des Produits par Ventes')
plt.xlabel('Ligne de Produit')
plt.ylabel('Ventes Totales')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Analyse de la relation entre le prix et les ventes
plt.figure(figsize=(12, 6))
plt.scatter(df['PRICEEACH'], df['SALES'], alpha=0.5)
plt.title('Relation entre Prix et Ventes')
plt.xlabel('Prix Unitaire')
plt.ylabel('Ventes Totales')
plt.grid()
plt.show()

# Code pour la préparation de la segmentation des produits
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Agrégation des données par catégorie de produit et taille de transaction
product_segmentation = file_path.groupby(['PRODUCTLINE', 'DEALSIZE']).agg({
    'QUANTITYORDERED': 'sum',
    'SALES': 'sum'
}).reset_index()

# Préparation des données pour le clustering
X = product_segmentation[['QUANTITYORDERED', 'SALES']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
product_segmentation['Cluster'] = kmeans.fit_predict(X_scaled)

# Fermer la connexion
conn.close()
