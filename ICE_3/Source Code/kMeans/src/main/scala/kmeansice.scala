import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

object kmeansice {

  def main(args: Array[String]): Unit = {

    val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")

    val sc=new SparkContext(sparkConf)

    // Turn off Info Logger for Consolexxx
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);
    // Load and parse the data
    val data = sc.textFile("data\\3D_spatial_network.txt")
    val parsedData_1 = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()

    //Look at how training data is!
    parsedData_1.foreach(f=>println(f))

    // Cluster the data into three classes and 50 iterations using KMeans
    val numClusters = 3
    val numIterations = 50
    val clusters_3 = KMeans.train(parsedData_1, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters_3.computeCost(parsedData_1)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    //Look at how the clusters are in training data by making predictions
    println("Clustering on training data for three classes and 50 iterations: ")
    clusters_3.predict(parsedData_1).zip(parsedData_1).foreach(f=>println(f._2,f._1))

    // Save and load model
    clusters_3.save(sc, "data\\Model3")
    val sameModel = KMeansModel.load(sc, "data\\Model3")

    // Cluster the data into four classes and 50 iterations using KMeans
    val numClusters_4 = 4
    val numIterations_4 = 50
    val clusters_4 = KMeans.train(parsedData_1, numClusters_4, numIterations_4)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE_4 = clusters_4.computeCost(parsedData_1)
    println("Within Set Sum of Squared Errors = " + WSSSE_4)

    //Look at how the clusters are in training data by making predictions
    println("Clustering on training data for 4 classes and 50 iterations: ")
    clusters_4.predict(parsedData_1).zip(parsedData_1).foreach(f=>println(f._2,f._1))

    // Save and load model
    clusters_4.save(sc, "data\\Model4")
    val sameModel_4 = KMeansModel.load(sc, "data\\Model4")

    // Cluster the data into three classes and 20 iterations using KMeans
    val numClusters_1 = 3
    val numIterations_1 = 20
    val clusters_1 = KMeans.train(parsedData_1, numClusters_1, numIterations_1)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE_1 = clusters_1.computeCost(parsedData_1)
    println("Within Set Sum of Squared Errors = " + WSSSE_1)

    //Look at how the clusters are in training data by making predictions
    println("Clustering on training data for 3 classes and 20 iterations: ")
    clusters_1.predict(parsedData_1).zip(parsedData_1).foreach(f=>println(f._2,f._1))

    // Save and load model
    clusters_1.save(sc, "data\\Model1")
    val sameModel_1 = KMeansModel.load(sc, "data\\Model1")

    // Cluster the data into four classes and 20 iterations using KMeans
    val numClusters_2 = 4
    val numIterations_2 = 20
    val clusters_2 = KMeans.train(parsedData_1, numClusters_2, numIterations_2)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE_2 = clusters_2.computeCost(parsedData_1)
    println("Within Set Sum of Squared Errors = " + WSSSE_2)

    //Look at how the clusters are in training data by making predictions
    println("Clustering on training data for 4 classes and 20 iterations: ")
    clusters_2.predict(parsedData_1).zip(parsedData_1).foreach(f=>println(f._2,f._1))

    // Save and load model
    clusters_2.save(sc, "data\\Model2")
    val sameModel_2 = KMeansModel.load(sc, "data\\Model2")
  }

}
