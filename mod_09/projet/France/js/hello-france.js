var w = 600;
var h = 600;
var r_max = 20
var on_mouseover_magnifier = 3; 
var dataset = [];
var x,y,pop, col;


//Create SVG element
var title = d3.select("body").append("h1").attr('id','title');
var test = d3.select("body").append("p").attr('id','test_zone');
var cor = d3.select("body").append("p").attr('id','gps');
var svg = d3.select("body")
            .append("svg")
            .attr("width", w)
            .attr("height", h);
var p = d3.select("body").append("p").attr('id','legende');

d3.tsv("data/france.tsv")
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

        pop = d3.scale.sqrt()
                    .domain(d3.extent(rows, function(row) { return row.population; }))
                    .range([1, r_max]);

        col = d3.scale.linear()
                    .domain(d3.extent(rows, function(row) { return row.densite; }))
                    .range(['blue', 'red']);
        draw(dataset);
    });

function draw(data) {
    svg.selectAll("circle").remove();
    svg.selectAll("circle")
        .data(data)
        .enter()
        .append("circle")
		.attr("r", function(d) { return pop(d.population); })
		.attr("cx", function(d) { return x(d.longitude); })
		.attr("cy", function(d) { return h-y(d.latitude); })
        .attr("fill", function(d) { return col(d.densite); })
		.on("mouseover", highlight)
		.on("mouseout", unhighlight);
	}

function highlight(d){

    d3.select(this)
    .attr("r", function(d) { return on_mouseover_magnifier*pop(d.population); });

	d3.select(this)
	.attr("fill", "yellow");

    d3.select("#title")
    .text(d.place);
    
	d3.select("#gps")
	.text("longitude=" + d.longitude + ", latitude=" + d.latitude);

    d3.select("#legende")
    .text(" Code postal=" + d.codePostal + ", population=" + d.population + ", densité=" + d.densite + " ... ");
    
 
}

function unhighlight(d){
	d3.select(this)
    .attr("r", function(d) { return pop(d.population); })
    .attr("fill", function(d) { return col(d.densite); });
}

function on_slide(evt, value) {
    d3.select('#slider3text').text(value);
    filtered = dataset.filter(function(d){ return d['population'] > value;});
    draw(filtered);
}
