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
