devtools::load_all()

# Klassen, die exportiert werden stehen in den `compboost_modules.cpp`
#
# - Deklaration von Funktionen in data.h
# - Implementieren von den Methoden in data.cpp



X = cbind(1, 1:10)
dobj = InMemoryData$new(X, "my-data")

dobj$getData()
dobj$getIdentifier()

path_binary = "binary-data-obj"
dobj$serialize(path_binary)
dobj$load(path_binary)


