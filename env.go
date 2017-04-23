package cuberl

import (
	"errors"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/gocube"
)

// Env is an anyrl.Env for cube states.
type Env struct {
	Creator   anyvec.Creator
	Objective Objective
	EpLen     int

	state    *State
	timestep int
}

// Reset generates a new state.
func (e *Env) Reset() (anyvec.Vector, error) {
	e.state = RandomStates(e.Objective, 1)[0]
	e.timestep = 0
	return e.vec(), nil
}

// Step takes a step in the environment.
//
// The input should be a one-hot move vector.
func (e *Env) Step(action anyvec.Vector) (obs anyvec.Vector, rew float64,
	done bool, err error) {
	if e.state == nil {
		err = errors.New("step Env: not initialized")
		return
	}
	move := gocube.Move(anyvec.MaxIndex(action))
	e.state, rew = e.state.Move(move)
	e.timestep++
	done = e.timestep == e.EpLen
	obs = e.vec()
	return
}

func (e *Env) vec() anyvec.Vector {
	return CubeVector(e.Creator, &e.state.Cube)
}
