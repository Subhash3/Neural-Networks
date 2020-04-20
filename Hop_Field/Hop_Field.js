class HopFieldNetwork{
    constructor(N){
        this.N = N
        this.weight_matrix = new Matrix(N, N)
        this.isNormalized = false
        // this.weight_matrix.display()
    }

    learn_input(X){
        X = Matrix.arrayToMatrix(X)
        var X_transpose = X.transpose()
        var dot_product = X_transpose.multiply(X)
        this.weight_matrix.add(dot_product)
        this.weight_matrix.display()
        this.make_diagonal_zero()
    }

    make_diagonal_zero(){
        for(var i = 0; i < this.N; i++){
            this.weight_matrix.table[i, i] = 0
        }
    }

    normalise(){
        if(! this.isNormalized){
            for(var i = 0; i < this.weight_matrix.table.rows; i++){
                for(var j = 0; j < this.weight_matrix.table.cols; j++){
                    this.weight_matrix.table[i][j] /= N
                }
            }
            this.isNormalized = true
        }
    }

    predict_one_val(X, i){
        this.normalise()
        X = Matrix.arrayToMatrix(X)
        ith_column_of_weight_matrix = this.weight_matrix.get_column(i)
        net = X.multiply(ith_column_of_weight_matrix)
        val = X.table[i]
        X.table[i] = this.activation(net, val)
        self.steps += 1
        
        return X
    }

    predict_one_step(X){
        for(var i = 0; i < this.N; i++){
            X = predict_one_val(X, i)
        }
        return X
    }

    predict(X){
        prev_inp = null
        this.steps = 0

        while(true){
            X = this.predict_one_step(X)

            if(prev_inp == X){
                return X
            }
            prev_inp = X
        }
    }

    activation(net, val){
        if(net > 0){
            return 1
        }
        else if(net == 0){
            return val
        }
        else{
            return -1
        }
    }

}