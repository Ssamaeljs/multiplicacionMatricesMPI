Para compilar se utiliza: mpicc ejemplo.c -o "Nombre del Objeto"
Para ejecutarlo se utiliza: mpirun "Nombre del Objeto"
Si se desea ejecutarlo en otro nodo : mpirun -H "nodo o dirección IP del nodo" "Nombre del Objeto"
Si se realiza un archivo de hosts se utiliza: mpirun -hostfile "Nombre del Objeto"
