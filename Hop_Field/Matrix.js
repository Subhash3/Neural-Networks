class Matrix{
    constructor(rows, cols){
        this.rows = rows
        this.cols = cols
        this.table = []

        for(var i = 0; i < this.rows; i++){
            var row = []
            for(var j = 0; j < this.cols; j++){
                row.push(1)
            }
            this.table.push(row)
        }
    }

    randomize(){
        for(var i = 0; i < this.rows; i++){
            for(var j = 0; j < this.cols; j++){
                this.table[i][j] = Math.round(Math.random()*10, 2)
            }
        }
    }

    display() {
        console.table(this.table)
    }

    transpose(){
        var new_Matrix = new Matrix(this.cols, this.rows)
        for(var j = 0; j < this.cols; j++){
            for(var i = 0; i < this.rows; i++){
                new_Matrix.table[j][i] = this.table[i][j]
            }
        }
        return new_Matrix
    }

    add(n){
        if(n instanceof Matrix){
            // console.log("Provided arg is a Matrix")
            if(this.rows != n.rows || this.cols != n.cols ){
                return false;
            }
            for(var i = 0; i < this.rows; i++){
                for(var j = 0; j < this.cols; j++){
                    this.table[i][j] += n.table[i][j]
                }
            }
            return true;
        }
        else{
            // console.log("Provided arg is a Scalar")
            for(var i = 0; i < this.rows; i++){
                for(var j = 0; j < this.cols; j++){
                    this.table[i][j] += n.table[i][j]
                }
            }
            return true;
        }
    }

    multiply(n){
        if(n instanceof Matrix){
            if(this.cols != n.rows){
                console.error(`Number of columns of the first matrix must be equal to rows of the second column`)
                return
            }
            var result = new Matrix(this.rows, n.cols)
            for(var i = 0; i < result.rows; i++){
                for(var j = 0; j < result.cols; j++){
                    result.table[i][j] = 0
                    for(var k = 0; k < this.cols; k++){
                        result.table[i][j] += this.table[i][k] * n.table[k][j]
                    }
                }
            }
            return result
        }
        else{
            for(var i = 0; i < this.rows; i++){
                for(var j = 0; j < this.cols; j++){
                    this.table[i][j] *= n
                }
            }
        }
    }

    static arrayToMatrix(arr){
        var m = new Matrix(1, arr.length)
        for(var i = 0; i < m.rows; i++){
            for(var j = 0; j < m.cols; j++){
                m.table[i][j] = arr[j]
            }
        }

        return m
    }

    get_row(index){
        index -= 1
        var row = new Matrix(1, this.cols)
        for(var i = 0; i < row.rows; i++){
            for(var j = 0; j < row.cols; j++){
                row.table[i][j] = this.table[index][j]
            }
        }
        return row
    }

    get_column(index){
        index -= 1

        var column = new Matrix(this.rows, 1)
        for(var i = 0; i < column.rows; i++){
            for(var j = 0; j < column.cols; j++){
                column.table[i][j] = this.table[i][index]
            }
        }
        return column
    }
}