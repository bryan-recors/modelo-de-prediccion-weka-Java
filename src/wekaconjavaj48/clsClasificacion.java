/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaconjavaj48;

import java.io.IOException;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource; //permite la conversion 
import weka.classifiers.trees.J48; //importar un clasificador
import weka.core.Instance; 
import weka.core.Instances;
//para evaluar el modelo
import weka.classifiers.Evaluation;
 import java.util.Random;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;

/**
 *
 * @author bryan
 */
public class clsClasificacion {
    String resultado = "";
    public String clasificar(
        Float duracion, 
        Float incremento1A, 
        Float incremento2A, 
        Float incremento3A, 
        String ajusteCostoVida, 
        Float horasTrabajo, 
        String pension, 
        Float pagoEnEspera, 
        Float diferencialDeCambio, 
        String subsidioEducacion, 
        Float diasFeriados, 
        String vacaciones, 
        String asistenciaDiscapacidad, 
        String contribucionPlanDental, 
        String ayudaDuelo, 
        String contribucionPlanSalud ) throws Exception {
        //**********************************cargar el dataset ************************************
        DataSource source = new DataSource("src/wekaconjavaj48/labor.arff"); //Indicamos el dataset
        Instances data = source.getDataSet(); //realizamos una isntancia con el dataset
        System.out.println(data.toString()); //imprimo el dataset
        
        //**********************************clasificar con el dataset *****************************
         System.out.println("\n=====Clasificacion Algotirmo J48 =====");
        data.setClassIndex(16); //numero de la clases respecto a la que quiero clasificar en este datasetr el 16
        String[] options = new String[1]; //creara un string para las opcines
        options[0] = "-U"; //guarda la opcion -U 
        J48 j48 = new J48(); //crear un clasificador j48
        j48.setOptions(options); //le paso las opciones que estan como variables globales 
        j48.buildClassifier(data); //clasificamos mandando el dataset 
        System.out.println(j48.toString()); //se imprime los resultados de la clasificacion 
        
        //**********************************Evaluacion del modelo sumary  *****************************
        System.out.println("\n=====summary=====");
        //evaluar el modelo
        Classifier cls = new J48(); //genera un clasificador J48 
        cls.buildClassifier(data);  //construyo la clasificacion co el dataset
        Evaluation eval = new Evaluation(data); //evalua la clasificacion mandando el dataset
        int numFolds = 10; //numero de particiones de la data 
        Random random = new Random(1); 
        //aplicar una evaluacion crizada mandando como parametro el metodo de clasificacion, la data, numero de division de la data, el random y el object
        eval.crossValidateModel(cls, data, numFolds, random, new Object[] {}); 
        eval.evaluateModel(cls, data); //mando a evaluar el modelo con el metodo de clasificacion y la data
        System.out.println(eval.toSummaryString()); // imprime el resultado del sumary
  
        //**********************************Evaluacion del modelo accuracy  *****************************
        System.out.println("\n=====acurracy=====");
        System.out.println(eval.toClassDetailsString());
        
        //**********************************Evaluacion del modelo Matriz de confusion   *****************************
        //matriz de confusion 
        System.out.println("\nMATRIZ DE CONFUSION ALGORITMO j48");
        double[][] confusionMatrix = eval.confusionMatrix();
        System.out.println(eval.toMatrixString());
        
        //**********************************clasificar una nueva instancia con los datos del formulario   *****************************
        System.out.println("clasificar una nueva instancia ingresada a traves del formulario :");
        Instance instance = new DenseInstance(16);
        instance.setDataset(data);
        instance.setValue(0, duracion); //duracion
        instance.setValue(1, incremento1A); //incremento primer año
        instance.setValue(2, incremento2A); //incremento seundo año
        instance.setValue(3, incremento3A); //incremento tercer año
        instance.setValue(4, ajusteCostoVida); //costo de vida 
        instance.setValue(5, horasTrabajo); // horas de trabajo
        instance.setValue(6, pension); //pension
        instance.setValue(7, pagoEnEspera); //pago en espera
        instance.setValue(8, diferencialDeCambio); //diferencial de cambio
        instance.setValue(9, subsidioEducacion); // subcidio a la educacion
        instance.setValue(10, diasFeriados); // dias feriados
        instance.setValue(11, vacaciones); //vacaciones
        instance.setValue(12, asistenciaDiscapacidad); //asistencia por discapacidad
        instance.setValue(13, contribucionPlanDental); //contribucion al plan dental
        instance.setValue(14, ayudaDuelo); // ayuda por duelo
        instance.setValue(15, contribucionPlanSalud); //contribucion al plan de salud        

        double result = j48.classifyInstance(instance); //que clasifique la instancia 
        System.out.println("Resultado de clasificar la nueva instancia:" + result);
        resultado = "correctamente clasificado";
        return data.classAttribute().value((int)result);
        
    }
}
