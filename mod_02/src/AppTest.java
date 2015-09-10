package fr.atih.srim.ds.module02.mongodb;

import java.net.UnknownHostException;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

import com.mongodb.BasicDBObject;
import com.mongodb.DB;
import com.mongodb.DBCollection;
import com.mongodb.DBCursor;
import com.mongodb.Mongo;

/**
 * Unit test for simple App.
 */
public class AppTest extends TestCase {
	/**
	 * Create the test case
	 *
	 * @param testName
	 *            name of the test case
	 */
	public AppTest(String testName) {
		super(testName);
	}

	/**
	 * @return the suite of tests being tested
	 */
	public static Test suite() {
		return new TestSuite(AppTest.class);
	}

	/**
	 * Rigourous Test :-)
	 */
	public void testApp() {
		assertTrue(true);
	}
	
	public void testMongo01() throws UnknownHostException{
		
		System.out.println("Beginning the test ...");
		
		Mongo mongo = new Mongo("localhost", 27017);
		System.out.println("Connected to MongoDB running on localhost 27017 !");
		
		DB db = mongo.getDB("prefs"); // if database doesn't exists, MongoDB will create it. Equivalent to "use prefs" in command line mode
		DBCollection table = db.getCollection("location"); // if collection doesn't exists, MongoDB will create it. Equivalent to "db.location" in command line mode.
		
		BasicDBObject document = new BasicDBObject();
		document.put("name", "Raja Chiky");
		document.put("zip", "75014");
		table.insert(document);
		
		document = new BasicDBObject();
		document.put("name", "Sylvain Lefebre");
		document.put("zip", "75015");
		table.insert(document);
		
		document = new BasicDBObject();
		document.put("name", "Olivier Hermant");
		document.put("zip", "75005");
		table.insert(document);
		
		document = new BasicDBObject();
		document.put("name", "Matthieu Manceny");
		document.put("zip", "92100");
		table.insert(document);
		
		System.out.println("4 people inserted");
		
		System.out.println("Table count = " + table.count());
		
		BasicDBObject searchQuery = new BasicDBObject(); // Creating an empty search query
		
		DBCursor cursor = table.find(searchQuery);
		while (cursor.hasNext()) {
			System.out.println(cursor.next()); // Iterating and showing found people
		}
		
		
		
		
		System.out.println("Removing all inserted people ...");
		BasicDBObject removeQuery = new BasicDBObject(); // Creating an empty query for removing all
		table.remove(removeQuery);
		System.out.println("All removed !");
		
		System.out.println("Test finished !");
		
	}
	
	
	public void testMongo02() throws UnknownHostException{
		
		System.out.println("Beginning TP Mongo 02 ...");
		
		Mongo mongo = new Mongo("localhost", 27017);
		System.out.println("Connected to MongoDB running on localhost 27017 !");
		
		DB db = mongo.getDB("Librairie"); // if database doesn't exists, MongoDB will create it. Equivalent to "use prefs" in command line mode
		DBCollection table = db.getCollection("Medias"); // if collection doesn't exists, MongoDB will create it. Equivalent to "db.location" in command line mode.
		
		
		//document = ( { Type : "Book", Title : "Definitive Guide to MongoDB", ISBN : "987-1-4302-3051-9", Publisher : "Apress", Author: ["Membrey, Peter", "Plugge, Eelco", "Hawkins, Tim" ] } )

		BasicDBObject document = new BasicDBObject();
		document.put("Type", "Book");
		document.put("Title", "Definitive Guide to MongoDB");
		document.put("ISBN", "987-1-4302-3051-9");
		document.put("Publisher", "Apress");
		
		table.insert(document);
		
		document = new BasicDBObject();
		document.put("name", "Sylvain Lefebre");
		document.put("zip", "75015");
		table.insert(document);
		
		document = new BasicDBObject();
		document.put("name", "Olivier Hermant");
		document.put("zip", "75005");
		table.insert(document);
		
		document = new BasicDBObject();
		document.put("name", "Matthieu Manceny");
		document.put("zip", "92100");
		table.insert(document);
		
		System.out.println("4 people inserted");
		
		System.out.println("Table count = " + table.count());
		
		BasicDBObject searchQuery = new BasicDBObject(); // Creating an empty search query
		
		DBCursor cursor = table.find(searchQuery);
		while (cursor.hasNext()) {
			System.out.println(cursor.next()); // Iterating and showing found people
		}
		
		
		
		
		System.out.println("Removing all inserted people ...");
		BasicDBObject removeQuery = new BasicDBObject(); // Creating an empty query for removing all
		table.remove(removeQuery);
		System.out.println("All removed !");
		
		System.out.println("Test finished !");
		
	}

}
