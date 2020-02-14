package imageprocess

import (
	"fmt"
	"strconv"
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func Test_AgePredict(t *testing.T) {
	tfClient := NewTFClient("age_andy.pb")
	defer tfClient.Close()
	b := tfClient.ReadJPGFromPath("/go/src/app/testimg/bona.jpg")
	imgTensorFormat := tfClient.MakeTensorFromImageByte(b)
	feedsOutput := map[tf.Output]*tf.Tensor{
		tfClient.ModelGraph.Operation("input_1").Output(0): imgTensorFormat,
	}
	fetchOutput := []tf.Output{
		//	tfClient.ModelGraph.Operation("gender/Softmax").Output(0),
		tfClient.ModelGraph.Operation("age/Sigmoid").Output(0),
	}
	output := tfClient.GetResult(feedsOutput, fetchOutput)

	// gender
	/*func(Obj []float32) {
		if Obj[0] > Obj[1] {
			fmt.Println("woman")
		} else {
			fmt.Println("man")
		}

	}(output[0].Value().([][]float32)[0])
	*/
	// age
	func(Obj []float32) {
		value, _ := strconv.ParseFloat(fmt.Sprintf("%.2f", Obj[0]*70.0), 64)
		fmt.Println(value)
	}(output[0].Value().([][]float32)[0])
}
