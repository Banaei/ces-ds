package fr.paristech.telecom.ces.ie;

import java.io.InputStream;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args ) throws Exception
    {
    	InformationExtractor ie = new InformationExtractor();
    	String testFilePath = "src/main/resources/data/test.txt";
    	String filePath = "data/wikifirst.txt";
    	
    	ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
    	InputStream is = classLoader.getResourceAsStream(filePath);
    	
		ie.process(is );
        System.out.println( "End !" );
    }
}
