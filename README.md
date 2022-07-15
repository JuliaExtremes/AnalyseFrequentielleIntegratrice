# Analyse fréquentielle intégratrice

Ce répertoire contient les fichiers requis pour l'activité de transfert du 21 juillet 2022.




## Préparation : installer Julia et Jupyter 

### Étape 1 : installer Julia 1.7

Allez sur [https://julialang.org/downloads](https://julialang.org/downloads) et téléchargez la version stable actuelle, Julia 1.7.3, en utilisant la version correcte pour votre système d'exploitation (Linux, Mac, Windows, etc).

### Étape 2 : exécuter Julia

Après l'installation, assurez-vous que vous pouvez exécuter Julia. Sur certains systèmes, cela signifie rechercher le programme **Julia 1.7** installé sur votre ordinateur ; sur d'autres, cela signifie exécuter la commande `julia` dans un terminal.

### Étape 3 : installer Jupyter via IJulia

Pour cette activité, nous utiliserons les notebooks Jupyter pour rassembler, éditer et exécuter du code Julia. Après avoir installé la dernière version de Julia, vous pouvez installer l'extension Jupyter à l'aide des deux commandes suivantes à exécuter dans un terminal Julia :

```julia
using Pkg
Pkg.add("IJulia")
```

Répondez oui lorsqu'on vous demande si vous souhaitez utiliser conda pour installer Jupyter.

### Étape 4 : exécuter Jupyter et ouvrir un calepin

Par la suite, vous pourrez lancer Jupyter à l'aide des commandes suivantes :

```julia
using IJulia
notebook()
```

## Installation des librairies nécessaires

Après avoir lancé Julia, exécutez les commandes suivantes pour installer les librairies nécessaires.

```julia
using Pkg

# Ajout des librairie pour les données
Pkg.add(["CSV", "DataFrames", "NetCDF"])

# Ajout des librairies de statistique
Pkg.add(["Distributions", "Extremes", "Mamba", "StatsBase"])
Pkg.add(url = "https://github.com/JuliaExtremes/ErrorsInVariablesExtremes.jl")

# Ajout de la librairie d'affichage
Pkg.add("Gadfly")

# Ajout de librairies utiles
Pkg.add(["ProgressMeter", "Serialization"])
```


## Cloner le répertoire GitHub

### Option 1

Dans un terminal, exécuter la commande ```git clone https://github.com/JuliaExtremes/AnalyseFrequentielleIntegratrice.git``` dans le répertoire de votre choix. Ceci vous permettra d'obtenir une copie à jour du répertoire. 

### Option 2

Sur la page d'accueil, cliquez sur le bouton vert intitulé `Code`. Vous pourrez alors choisir vos options de téléchargement.


## Ouvrir un fichier .ipynb

1. Lancez Julia
2. Exécutez les commandes 
```julia
using IJulia
notebook()
```
3. Recherchez dans vos fichiers locaux le calepin Jupyter avec l'extension .ipynb.

