
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object confusionScala {

  def main(args: Array[String]) {


    val sparkConf = new SparkConf().setAppName("confusionScala").setMaster("Confusion Matrix")
    val sc = new SparkContext(sparkConf)


    val data = sc.textFile("/Volumes/Data/BigDataAnalytics/ICMP4/CS5542-Tutorial2B-SourceCode/image_classification_Linux_MacOS/input/sample.txt")
    val parsedData = data.map(line=>{line.split(",")})
    val parsedwords = parsedData.map(word=>(word(0),word(1)))
    parsedwords.foreach(println)

    val predictionAndActual =  parsedwords.map(firstword => (if (firstword._1.equals("man")) 0.0  else  1.0,

                                                             if (firstword._2.equals("man")) 0.0 else 1.0))


    val confusion_metrics = new MulticlassMetrics(predictionAndActual)
    val confusionMatrix = confusion_metrics.confusionMatrix
    println("Confusion Matrix:\n" )
    println(confusionMatrix)

    val tp = confusionMatrix(0, 0)   // True positive
    val fn = confusionMatrix(0, 1)   // False negative
    val fp = confusionMatrix(1, 0)   // False positive
    val tn = confusionMatrix(1, 1)   // True negative
    val total = tp+fn+fp+tn

    //Accuracy=(TP+TN)/total
    val accuracy = ((tp+tn)/total).toFloat
    println("\nAccuracy :" + accuracy )

    //Misclassification Rate=(FP+FN)/total
    val miscRate = ((fp+fn)/total).toFloat
    println("Misclassification Rate :" + miscRate )

    // True Positive Rate=TP/actual yes
    val truePositiveRate = (tp/(tp+fn)).toFloat
    println("True Positive Rate :" + truePositiveRate)

    //False Positive Rate:FP/actual no
    val falsePositiveRate = (fp/(fp+tn)).toFloat
    println("false Positive Rate :" + falsePositiveRate)

    //Specificity=TN/actual no
    val specificity = (tn/(fp+tn)).toFloat
    println("Sensitivity :" + specificity)

    //Precision=TP/predicted yes
    val precision = (tp/(fp+tp)).toFloat
    println("Precision :" + precision)

    //Prevalence=actual yes/total
    val prevalence = (tp+fn/(total)).toFloat
    println("Precision :" + prevalence)



  }

}