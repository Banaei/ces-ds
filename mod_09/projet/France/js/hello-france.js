var w = 600;
var h = 600;
var dataset = [];
var x,y;

//Create SVG element
var cor = d3.select("body").append("p").attr('id','gps');
var svg = d3.select("body")
            .append("svg")
            .attr("width", w)
            .attr("height", h);
var p = d3.select("body").append("p").attr('id','legende');

d3.tsv("data/france-short.tsv")
    .row(function (d, i) {
        return {
            codePostal: d["Postal Code"],
            inseeCode: d.inseecode,
            place: d.place,
            longitude: +d.x,
            latitude: +d.y,
            population: +d.population,
            densite: +d.density
        };
    })

.get(function(error, rows) {
        console.log("Loaded " + rows.length + " rows");
        if (rows.length > 0) {
            console.log("First row: ", rows[0]);
            console.log("Last  row: ", rows[rows.length-1]);
        }
        dataset = rows;
        x = d3.scale.linear()
                    .domain(d3.extent(rows, function(row) { return row.longitude; }))
                    .range([0, w]);

		y = d3.scale.linear()
                    .domain(d3.extent(rows, function(row) { return row.latitude; }))
                    .range([0, h]);
        draw();
    });

function draw() {
    svg.selectAll("rect")
        .data(dataset)
        .enter()
        .append("rect")
		.attr("width", 1)
        .attr("height", 1)
		.attr("x", function(d) { return x(d.longitude); })
		.attr("y", function(d) { return h-y(d.latitude); })
		.on("mouseover", highlight)
		.on("mouseout", unhighlight);

		;
	}

function highlight(d){
	d3.select(this)
	.attr("width", 6)
	.attr("height", 6)
	.attr("fill", "red");

	d3.select("#legende")
	.text(d.place + " Code postal=" + d.codePostal + ", population=" + d.population + ", densité=" + d.densite);
	
	d3.select("#gps")
	.text("longitude=" + d.longitude + ", latitude=" + d.latitude);
}

function unhighlight(d){
	d3.select(this)
	.attr("width", 1)
	.attr("height", 1)
	.attr("fill", "black");
}

