function x = reduceToLastNIndices(matrix, N)

if N >= length(matrix)
    x = matrix;
    disp('Error: N here is >= the length of the input matrix')
else
    x = matrix((length(matrix) - N):length(matrix));
end
end