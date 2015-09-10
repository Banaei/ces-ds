package fr.paristech.telecom.ces.ie;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class DataHolder {
	
	private SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
	private String title;
	private List<Date> datesList;
	private List<String> typesList;
	private List<String> locationList;
	
	public String getTitle() {
		return title;
	}
	public void setTitle(String title) {
		this.title = title;
	}
	public List<Date> getDatesList() {
		return datesList;
	}
	public void setDatesList(List<Date> datesList) {
		this.datesList = datesList;
	}
	public boolean hasData(){
		return (this.datesList!=null && this.datesList.size()>0) | (this.typesList!=null && this.typesList.size()>0);
	}
	public void addDate(Date date){
		if (this.datesList==null){
			this.datesList = new ArrayList<Date>();
		}
		this.datesList.add(date);
	}
	public void addType(String type){
		if (this.typesList==null){
			this.typesList = new ArrayList<String>();
		}
		this.typesList.add(type);
	}
	public void addLocation(String location){
		if (this.locationList==null){
			this.locationList = new ArrayList<String>();
		}
		this.locationList.add(location);
	}
	
	public String toString(){
		
		StringBuilder sb = new StringBuilder();
		
		sb.append(title).append("\n");

		if (datesList!=null && datesList.size()>0){
			sb.append("\thas date(s)\t");
			for (Date date : datesList) {
				sb.append(dateFormat.format(date)).append("\t");
			}
			sb.append("\n");
		}
		if (typesList!=null && typesList.size()>0){
			for (String type : typesList) {
				sb.append("\t type \t").append(type).append("\t");
			}
			sb.append("\n");
		}
		if (locationList!=null && locationList.size()>0){
			for (String location : locationList) {
				sb.append("\t located in \t").append(location).append("\t");
			}
			sb.append("\n");
		}
		
		sb.append("\n");
		
		return sb.toString();
	}
	
}
