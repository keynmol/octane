# Octane

A simple feedforward neural network gem that has the ability to bypass selected units during training/testing(used in dropout techniques, for example) and generate graphviz representation of the network.

## Installation

Add this line to your application's Gemfile:

    gem 'octane'

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install octane

## Usage

### Training and testing

```ruby
require 'octane'

include Octane # to make things clearer

net = Network.new
net.add_layer(2) # two inputs
net.add_layer(5) # five hidden neurons
net.add_layer(1) # one output

function = ->(x,y) { Math.cos(x+y) } # let's try to approximate cosine of sum of two variables 

test_dataset = Dataset.new
train_dataset = Dataset.new

# create two datasets, 100 and 10000 samples respectively
[test_dataset, train_dataset].zip([100, 10000]).each do |dataset, samples| 
    samples.times do 
        input = [rand, rand]
        dataset.add_sample(input, function[*input]) 
    end
end

puts "Test error before training: #{net.test(test_dataset)}"
net.train(train_dataset, epochs: 1000) # use stochastic gradient descent for 1000 iterations
puts "Test error after training: #{net.test(test_dataset)}"
```

You will see something like this:

	Test error before training: 0.04733634460348299
	Test error after training: 0.012294661498887489

### Plotting the network and disabling neurons
Make sure you have RMagick installed to run this example without errors. If not, just use 
```ruby
require 'octane'
require 'RMagick'

include Magick
include Octane

# simple plotting function. The only thing that matters here is the Octane::Network#graphviz method
def plot(n, name=nil, params=nil)
	params=[{}] unless params
	name="temp.png" unless name
	File.open("nn.dot","w") {|f|
		f.write(n.graphviz(*params))
	}


	`dot nn.dot -Tpng > #{name}`

	cat = ImageList.new(name)
	cat.display
end

net=Network.new
net.add_layer(2) # two inputs
net.add_layer(5) # five hidden neurons
net.add_layer(1) # one output

plot(net)

```

This will show you a picture that looks roughly like this:

![Network](http://i.imgur.com/OXorGtg.png)

### Dropout
Building up on previous examples, here's how one can disable and enable certain neurons in a network:
```ruby
# disable
net.hidden_layer(0).disable(3)

# enable
net.hidden_layer(0).enable(3)
```
Disable neuron will be ignored during both forward pass and backpropagation, effectively reducing the size of trained network.
Disabled units are also clearly shown on a plot:
![Network with disabled units](http://i.imgur.com/m8maKnD.png)

## Future
Further performance optimisations, more hooks to the training process, extract different learning modes into separate classes, add softmax layers, simplify API, make plotting more customisable.

## Contributing

1. Fork it
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request
