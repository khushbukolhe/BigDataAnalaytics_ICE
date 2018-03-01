

import org.apache.spark.{SparkContext, SparkConf}

object SparkWordCount {

  def main(args: Array[String]) {

    System.setProperty("hadoop.home.dir","C:\\winutils");

    val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")

    val sc=new SparkContext(sparkConf)

    val input=sc.parallelize(Array("United", "States", "Incident", "Separated", "Unified", "Investments", "Board"))

    val b= input.groupBy(word=>word.charAt(0))

    b.saveAsTextFile("output")

    val output=b.collect()

  }

}

