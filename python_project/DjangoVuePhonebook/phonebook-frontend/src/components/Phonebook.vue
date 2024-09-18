<template>
  <div>
    <h2>联系人列表</h2>
    <ul>
      <li v-for="(contact, index) in contacts" :key="index">
        {{ contact.name }} - {{ contact.phone }}
        <button @click="editContact(index)">编辑</button>
        <button @click="deleteContact(index)">删除</button>
      </li>
    </ul>

    <h2>{{ isEditing ? '编辑联系人' : '添加联系人' }}</h2>
    <form @submit.prevent="saveContact">
      <input v-model="newContact.name" placeholder="姓名" required />
      <input v-model="newContact.phone" placeholder="电话" required />
      <button type="submit">{{ isEditing ? '保存' : '添加' }}</button>
    </form>
  </div>
</template>

<script>
export default {
  data() {
    return {
      contacts: [
        { name: '张三', phone: '123456789' },
        { name: '李四', phone: '987654321' }
      ],
      newContact: { name: '', phone: '' },
      isEditing: false,
      currentIndex: null
    }
  },
  methods: {
    saveContact() {
      if (this.isEditing) {
        this.contacts[this.currentIndex] = this.newContact
        this.isEditing = false
        this.currentIndex = null
      } else {
        this.contacts.push(this.newContact)
      }
      this.newContact = { name: '', phone: '' }
    },
    editContact(index) {
      this.newContact = { ...this.contacts[index] }
      this.isEditing = true
      this.currentIndex = index
    },
    deleteContact(index) {
      this.contacts.splice(index, 1)
    }
  }
}
</script>

<style scoped>
h2 {
  color: #42b983;
}

ul {
  list-style-type: none;
  padding: 0;
}

li {
  margin-bottom: 10px;
}

button {
  margin-left: 10px;
}
</style>
