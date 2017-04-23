package main

import (
	"flag"
	"log"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/cuberl"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

func main() {
	var stepSize float64
	var batchSize int
	var netFile string
	var hiddenSize int
	var epLen int
	var objective cuberl.Objective
	var cgIters int
	var layers int

	flag.Float64Var(&stepSize, "step", 0.01, "TRPO step size")
	flag.IntVar(&cgIters, "cgiters", 10, "CG iterations for TRPO")
	flag.IntVar(&batchSize, "batch", 10, "experience batch size")
	flag.IntVar(&hiddenSize, "hidden", 128, "LSTM state size for new agents")
	flag.IntVar(&layers, "layers", 1, "number of LSTM layers")
	flag.StringVar(&netFile, "net", "out_net", "network file path")
	flag.IntVar(&epLen, "len", 20, "episode length")
	flag.Var(&objective, "objective", cuberl.ObjectiveUsage)

	flag.Parse()

	creator := anyvec32.CurrentCreator()

	var policy anyrnn.Block
	if err := serializer.LoadAny(netFile, &policy); err != nil {
		log.Println("Creating new network...")
		policy = cuberl.NewPolicy(creator, layers, hiddenSize)
	} else {
		log.Println("Loaded network.")
	}

	envs := make([]anyrl.Env, batchSize)
	for i := range envs {
		envs[i] = &cuberl.Env{
			Creator:   creator,
			Objective: objective,
			EpLen:     epLen,
		}
	}

	trpo := &anyrl.TRPO{
		NaturalPG: anyrl.NaturalPG{
			Policy:      policy,
			Params:      anynet.AllParameters(policy),
			ActionSpace: anyrl.Softmax{},
			Iters:       cgIters,
		},
		TargetKL: stepSize,
	}

	r := rip.NewRIP()
	var batchIdx int
	for !r.Done() {
		batch, err := anyrl.RolloutRNN(creator, policy, anyrl.Softmax{}, envs...)
		if err != nil {
			panic(err)
		}
		log.Printf("batch %d: reward=%v", batchIdx, batch.MeanReward(creator))
		batchIdx++
		if r.Done() {
			break
		}
		trpo.Run(batch).AddToVars()
	}

	log.Println("Saving...")
	if err := serializer.SaveAny(netFile, policy); err != nil {
		essentials.Die(err)
	}
}
