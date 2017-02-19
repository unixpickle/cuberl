package cuberl

import (
	"math/rand"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anys2s"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/gocube"
)

// Samples creates sequence-to-sequences samples that are
// meant to be used in conjunction with anynet.DotCost{}.
func Samples(agent anyrnn.Block, start []*State, steps int, baseline float64) *anys2s.Batch {
	if steps == 0 {
		panic("must take at least one step")
	}

	state := append([]*State{}, start...)
	blockState := agent.Start(len(start))
	cr := agentCreator(agent)

	var ins, outs []*anyseq.Batch
	present := make([]bool, len(start))
	for i := range present {
		present[i] = true
	}

	moveHistory := [][]int{}
	totalRewards := make([]float64, len(state))
	for i := 0; i < steps; i++ {
		cubeIns := make([]anyvec.Vector, len(state))
		for j, s := range state {
			cubeIns[j] = CubeVector(cr, &s.Cube)
		}
		inVec := cr.Concat(cubeIns...)
		ins = append(ins, &anyseq.Batch{Packed: inVec, Present: present})

		res := agent.Step(blockState, inVec)
		blockState = res.State()

		var moves []int
		for j, s := range state {
			moveIdx := sampleMove(res.Output().Slice(j*NumActions, (j+1)*NumActions))
			var rew float64
			state[j], rew = s.Move(gocube.Move(moveIdx))
			totalRewards[j] += rew
			moves = append(moves, moveIdx)
		}
		moveHistory = append(moveHistory, moves)
	}

	for i := range totalRewards {
		totalRewards[i] -= baseline
	}

	rewardVec := cr.MakeVectorData(cr.MakeNumericList(totalRewards))
	for _, moves := range moveHistory {
		// Turn the move choices into a mapping table.
		for i := range moves {
			moves[i] += NumActions * i
		}

		outVec := cr.MakeVector(NumActions * len(state))
		mapper := cr.MakeMapper(outVec.Len(), moves)
		mapper.MapTranspose(rewardVec, outVec)
		outs = append(outs, &anyseq.Batch{Packed: outVec, Present: present})
	}

	return &anys2s.Batch{
		Inputs:  anyseq.ConstSeq(cr, ins),
		Outputs: anyseq.ConstSeq(cr, outs),
	}
}

func sampleMove(logProbs anyvec.Vector) int {
	anyvec.Exp(logProbs)

	data64 := make([]float64, logProbs.Len())
	switch data := logProbs.Data().(type) {
	case []float32:
		for i, x := range data {
			data64[i] = float64(x)
		}
	case []float64:
		copy(data64, data)
	}

	num := rand.Float64()
	for i, x := range data64 {
		num -= x
		if num < 0 {
			return i
		}
	}

	// Rounding errors might bring us here.
	return len(data64) - 1
}
