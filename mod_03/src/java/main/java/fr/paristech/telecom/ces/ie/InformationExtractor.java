package fr.paristech.telecom.ces.ie;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class InformationExtractor {

	String year = "[0-9]*";
	String year4 = "[0-9]{4}";
	String day = "((0?[1-9]|[12][0-9]|3[12]),?)";
	String month = "(January|February|March|April|May|June|July|August|September|October|November|December)";
	String date_1 = "(" + month + " " + day + " " + year + ")";
	String date_2 = "(" + day + " " + month + " " + year + ")";
	String date_3 = "from " + year4;
	String date_4 = "in " + year4;
	String date_regex = date_1 + "|" + date_2 + "|" + date_3 + "|" + date_4;
	
	DateFormat df_1_1 = new SimpleDateFormat("MMMMM dd, yyyy", Locale.ENGLISH);
	DateFormat df_1_2 = new SimpleDateFormat("MMMMM dd yyyy", Locale.ENGLISH);
	DateFormat df_1_3 = new SimpleDateFormat("MMMMM dd", Locale.ENGLISH);
	DateFormat df_2_1 = new SimpleDateFormat("dd MMMMM yyyy", Locale.ENGLISH);
	DateFormat df_3_1 = new SimpleDateFormat("yyyy", Locale.ENGLISH);
	
	Pattern date_pattern = Pattern.compile(date_regex);
	
	String typeRegex = "is .+";
	Pattern type_pattern = Pattern.compile(typeRegex);
	
	String locationRegex = "[A-Z][a-z]+ in [A-Z][a-z]+";
	Pattern location_pattern = Pattern.compile(locationRegex);
	
	
	public void process(InputStream  is) throws Exception {
		List<DataHolder> dataList = new ArrayList<DataHolder>();
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		
		String title = "";
		String line;
		while ((line = br.readLine()) != null) {
			if (title.equalsIgnoreCase("")) {
				title = line;
				continue;
			}
			if (line.trim().length() == 0) {
				title = "";
				continue;
			}
			DataHolder dh = new DataHolder();
			dh.setTitle(title);
			
			matchDate(line, dh);
			findtype(line, dh);
			findLocation(line, dh);
			
			
			dataList.add(dh);
		}
		br.close();
		
		printList(dataList);
	}
	
	private void findLocation(String line, DataHolder dh){
		Matcher matcher = location_pattern.matcher(line);
		while (matcher.find()) {
			String match = matcher.group();
			match = trimAfterWord(match, "in");
			dh.addLocation(match);
		}
	}
	
	private void findtype(String line, DataHolder dh){
		Matcher matcher = type_pattern.matcher(line);
		while (matcher.find()) {
			String match = matcher.group();
			match = trimBeforeWord(match, "that");
			match = trimBeforeWord(match, ",");
			dh.addType(match.substring(2));
		}
	}
	
	private String trimBeforeWord(String s, String word){
		int i = s.indexOf(word);
		if (i==-1){
			return s;
		} else {
			return s.substring(0, i);
		}
	}

	private String trimAfterWord(String s, String word){
		int i = s.indexOf(word);
		if (i==-1){
			return s;
		} else {
			return s.substring(i+word.length(), s.length());
		}
	}

	
	private void printList(List<DataHolder> dataList){
		for (DataHolder dh : dataList) {
			if (dh.hasData()){
				System.out.println(dh.toString());
			}
		}
	}

	private void matchDate(String line, DataHolder dh) throws ParseException {
		Matcher matcher = date_pattern.matcher(line);
		while (matcher.find()) {
			String match = matcher.group();
			Date date = null;
			if (match.matches(date_1)){
				try {
					date = df_1_1.parse(match);
				} catch (ParseException e) {
				}
				if (date==null){
					try {
						date = df_1_2.parse(match);
					} catch (ParseException e) {
					}
				}
				if (date==null){
					try {
						date = df_1_3.parse(match);
					} catch (ParseException e) {
					}
				}
				
			} else if(match.matches(date_2)){
				try {
					date = df_2_1.parse(match);
				} catch (ParseException e) {
				}
				
			} else if(match.matches(date_3)){
				try {
					date = df_3_1.parse(match.substring(4));
				} catch (ParseException e) {
				}
				
			} else if(match.matches(date_4)){
				try {
					date = df_3_1.parse(match.substring(2));
				} catch (ParseException e) {
				}
			}
			if (date!=null){
				dh.addDate(date);
			}
		}
	}

}
