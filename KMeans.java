import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.Scanner;
import java.util.Vector;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

class Point implements WritableComparable {
	public double x;
	public double y;

	public void readFields(DataInput in) throws IOException {
		this.x = in.readDouble();
		this.y = in.readDouble();
	}

	public void write(DataOutput out) throws IOException {
		out.writeDouble(this.x);
		out.writeDouble(this.y);
	}

	public int compareTo(Object o) {
		/*
		 * How do you compare 2 Points? Compare the x components first; if equal,
		 * compare the y components.
		 */
		Point p = (Point) o;
		int xEquality = Double.compare(this.x, p.x);
		if (xEquality != 0) {
			return xEquality;
		}
		return Double.compare(this.y, p.y);
	}

	@Override
	public String toString() {
		return x + "," + y;
	}
}

public class KMeans {
	static Vector<Point> centroids = new Vector<Point>(100);

	public static class AvgMapper extends Mapper<Object, Text, Point, Point> {

		@Override
		protected void setup(Context context)
				throws IOException, InterruptedException {
			super.setup(context);

			URI[] paths = context.getCacheFiles();
			Configuration conf = context.getConfiguration();
			FileSystem fs = FileSystem.get(conf);
			BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(paths[0]))));
			String aLine;
			while ((aLine = reader.readLine()) != null) {
				Scanner s = new Scanner(aLine).useDelimiter(",");
				Point p = new Point();
				p.x = s.nextDouble();
				p.y = s.nextDouble();
				s.close();
				centroids.add(p);
			}
		}

		@Override
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			Scanner s = new Scanner(value.toString()).useDelimiter(",");
			double px = s.nextDouble();
			double py = s.nextDouble();
			s.close();
			Point p = new Point();
			p.x = px;
			p.y = py;

			Point closestCentroid = centroids.firstElement();
			double minimumDistance = Double.MAX_VALUE;
			for (Point c : centroids) {
//				System.out.println("Comparing centroid " + c + " with point " + p);
				double existingDistance = calculateEuclideanDistance(p, c);
				if (Double.compare(existingDistance, minimumDistance) < 0) {
					minimumDistance = existingDistance;
					closestCentroid = c;
//					System.out.println("Closest centroid of Point " + p + " +  is " + closestCentroid
//							+ " with distance " + minimumDistance);
				}
			}

			context.write(closestCentroid, p);
		}

		private double calculateEuclideanDistance(Point p, Point c) {
			double sumOfSquares = (c.y - p.y) * (c.y - p.y) + (c.x - p.x) * (c.x - p.x);
			return Math.sqrt(sumOfSquares);
		}
	}

	public static class AvgReducer extends Reducer<Point, Point, Point, Object> {
		@Override
		public void reduce(Point centroid, Iterable<Point> clusterPoints, Context context)
				throws IOException, InterruptedException {

			System.out.println("Reducer...");
			/*
			 * reduce ( c, points ): count = 0 sx = sy = 0.0 for p in points count++ sx +=
			 * p.x sy += p.y c.x = sx/count c.y = sy/count emit(c,null)
			 */
			int count = 0;
			double sumX = 0.0;
			double sumY = 0.0;
			for (Point p : clusterPoints) {
				sumX += p.x;
				sumY += p.y;
				count++;
			}
			Point newCentroid = new Point();
			newCentroid.x = sumX / count;
			newCentroid.y = sumY / count;
			context.write(newCentroid, null);
		}
	}

	public static void main(String[] args) throws Exception {
		if (args.length != 3) {
			String errorMsg = "Invoke program with 3 parameters - the input file, the centroids file, the output directory";
			System.err.println(errorMsg);
		}

		Job job = Job.getInstance();
		job.setJarByClass(KMeans.class);
		job.setJobName("KMeans");
		// job.setOutputKeyClass(Point.class);
//		job.setOutputValueClass(null);
		job.setOutputFormatClass(TextOutputFormat.class);
		job.setMapperClass(KMeans.AvgMapper.class);
		job.setReducerClass(KMeans.AvgReducer.class);
		job.addCacheFile(new URI(args[1]));
		job.setMapOutputKeyClass(Point.class);
		job.setMapOutputValueClass(Point.class);

		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[2]));

		int returnValue = job.waitForCompletion(true) ? 0 : 1;
		if (job.isSuccessful()) {
			System.out.println("Job was successful! :)");
		} else if (!job.isSuccessful()) {
			System.out.println("Job was not successful. :(");
		}

		System.exit(returnValue);
	}
}
